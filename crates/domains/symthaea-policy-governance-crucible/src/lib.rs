// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adversarial qualification crucible for assurance-policy governance.

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
    ManifestSigningAuthorityPolicy, SigningAuthorityGovernance, SigningAuthorityTransition,
};
use symthaea_evidence_anchored_governed_readiness::assess_anchored_governed_readiness;
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
use symthaea_evidence_verifier_diversity::{
    VerifierDiversityPolicy, VerifierDiversityRequirement, VerifierFaultDomainProfile,
};
use symthaea_formal_safety::{
    EvidenceKind, ProofObligation, SafetyCase, SafetyEvidenceReceipt, StrictSafetyCaseStatus,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyGovernanceCrucibleStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyGovernanceScenario {
    pub scenario_id: String,
    pub expected: StrictSafetyCaseStatus,
    pub observed: StrictSafetyCaseStatus,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyGovernanceCrucibleReport {
    pub schema_version: String,
    pub status: PolicyGovernanceCrucibleStatus,
    pub scenarios: Vec<PolicyGovernanceScenario>,
}

impl PolicyGovernanceCrucibleReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

fn scenario(
    scenario_id: &str,
    expected: StrictSafetyCaseStatus,
    observed: StrictSafetyCaseStatus,
) -> PolicyGovernanceScenario {
    PolicyGovernanceScenario {
        scenario_id: scenario_id.into(),
        expected,
        observed,
        passed: expected == observed,
    }
}

fn safety_case() -> SafetyCase {
    let mut case = SafetyCase::new("policy-governance-crucible");
    case.add_obligation(
        ProofObligation::new("composite assurance claim", EvidenceKind::Test)
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

fn policy_binding(
    kind: AssurancePolicyKind,
    revision: u64,
    override_digest: Option<&str>,
) -> AssurancePolicyBinding {
    AssurancePolicyBinding {
        kind,
        policy_id: format!("policy:{}", kind.code()),
        schema_version: "1".into(),
        content_digest: override_digest
            .map(str::to_owned)
            .unwrap_or_else(|| format!("blake3:{}-r{revision}", kind.code())),
        evidence_refs: vec![format!("review:{}:r{revision}", kind.code())],
    }
}

fn manifest(
    case: &SafetyCase,
    revision: u64,
    predecessor: Option<String>,
    weakened_verifier_policy: bool,
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
            .map(|kind| {
                let weakened = weakened_verifier_policy
                    && kind == AssurancePolicyKind::VerifierDiversity;
                policy_binding(
                    kind,
                    revision,
                    weakened.then_some("blake3:verifier-diversity-weakened"),
                )
            })
            .collect(),
        evidence_refs: vec![format!("review:manifest-r{revision}")],
    }
}

fn signature(
    manifest: &AssurancePolicyManifest,
    signer: &str,
    key: &str,
) -> ManifestSignatureVerificationReceipt {
    ManifestSignatureVerificationReceipt {
        receipt_id: format!("sig:r{}:{signer}:{key}", manifest.revision),
        manifest_digest: manifest.manifest_digest(),
        signer_ref: signer.into(),
        key_ref: key.into(),
        signature_algorithm: "ed25519".into(),
        signature_ref: format!("signature:r{}", manifest.revision),
        verified_by_ref: "verifier:independent-signature".into(),
        verification_ref: format!("verification:r{}", manifest.revision),
        verified_at_ms: 500 + manifest.revision,
        evidence_refs: vec![format!("audit:signature:r{}", manifest.revision)],
    }
}

fn standard_lineage(case: &SafetyCase, revisions: u64) -> Vec<SignedAssurancePolicyRevision> {
    let mut lineage = Vec::new();
    let mut predecessor = None;
    for revision in 1..=revisions {
        let current = manifest(case, revision, predecessor.clone(), false);
        predecessor = Some(current.manifest_digest());
        lineage.push(SignedAssurancePolicyRevision {
            signature_receipt: signature(&current, "signer:a", "key:a1"),
            manifest: current,
        });
    }
    lineage
}

fn rotated_lineage(case: &SafetyCase) -> Vec<SignedAssurancePolicyRevision> {
    let r1 = manifest(case, 1, None, false);
    let r2 = manifest(case, 2, Some(r1.manifest_digest()), false);
    vec![
        SignedAssurancePolicyRevision {
            signature_receipt: signature(&r1, "signer:a", "key:a1"),
            manifest: r1,
        },
        SignedAssurancePolicyRevision {
            signature_receipt: signature(&r2, "signer:b", "key:b1"),
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
            sample_id: "t1".into(),
            wall_time_ms: 1_000,
            monotonic_time_ms: 5_000,
            clock_source: "ptp-a".into(),
            clock_domain: "domain-a".into(),
            epoch_ref: "epoch-a".into(),
            clock_uncertainty_ms: 10,
            evidence_refs: vec!["clock:t1".into()],
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
        .expect("valid deployment receipt")
}

fn policy_receipts(
    case: &SafetyCase,
    manifest: &AssurancePolicyManifest,
    signature_receipt: &ManifestSignatureVerificationReceipt,
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
            manifest,
            signature_receipt,
            format!("policy-scope:{id}"),
        )
        .expect("valid policy receipt")
    })
    .collect()
}

fn verifier_profiles() -> Vec<VerifierFaultDomainProfile> {
    [
        ("verifier:a", "a"),
        ("verifier:b", "b"),
    ]
    .into_iter()
    .map(|(verifier, suffix)| VerifierFaultDomainProfile {
        verifier_ref: verifier.into(),
        organization_domain: format!("org:{suffix}"),
        review_process_domain: format!("process:{suffix}"),
        toolchain_domain: format!("tool:{suffix}"),
        evidence_source_domain: format!("source:{suffix}"),
        evidence_refs: vec![format!("profile:{verifier}")],
    })
    .collect()
}

fn verifier_policy(case: &SafetyCase, minimum: usize) -> VerifierDiversityPolicy {
    VerifierDiversityPolicy {
        schema_version: "1".into(),
        policy_id: format!("diversity-{minimum}"),
        requirements: vec![VerifierDiversityRequirement {
            obligation_key: case.obligations[0].stable_key(),
            minimum_distinct_verifiers: minimum,
            minimum_organization_domains: minimum,
            minimum_review_process_domains: minimum,
            minimum_toolchain_domains: minimum,
            minimum_evidence_source_domains: minimum,
            evidence_refs: vec!["review:diversity".into()],
        }],
        evidence_refs: vec!["policy:diversity".into()],
    }
}

fn atomic_policy(case: &SafetyCase) -> AtomicCoveragePolicy {
    AtomicCoveragePolicy {
        schema_version: "1".into(),
        policy_id: "atomic-v1".into(),
        facets: vec![AtomicEvidenceFacet {
            facet_id: "facet:a".into(),
            obligation_key: case.obligations[0].stable_key(),
            controlled_claim: "facet a".into(),
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
            binding_id: "b1".into(),
            receipt_id: "r1".into(),
            facet_id: "facet:a".into(),
            facet_evidence_ref: "result:r1".into(),
            facet_evidence_digest: "blake3:facet-r1".into(),
            rationale_ref: "rationale:b1".into(),
        },
        FacetEvidenceBinding {
            binding_id: "b2".into(),
            receipt_id: "r2".into(),
            facet_id: "facet:a".into(),
            facet_evidence_ref: "result:r2".into(),
            facet_evidence_digest: "blake3:facet-r2".into(),
            rationale_ref: "rationale:b2".into(),
        },
    ]
}

fn signing_governance(transitions: Vec<SigningAuthorityTransition>) -> SigningAuthorityGovernance {
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
        transitions,
    }
}

fn rotation_to_b() -> SigningAuthorityTransition {
    SigningAuthorityTransition {
        transition_id: "rotation:r2".into(),
        effective_revision: 2,
        from_signer_ref: "signer:a".into(),
        from_key_ref: "key:a1".into(),
        to_signer_ref: "signer:b".into(),
        to_key_ref: "key:b1".into(),
        authorization_ref: "authorization:r2".into(),
        independent_verification_ref: "verification:rotation:r2".into(),
        evidence_refs: vec!["audit:rotation:r2".into()],
    }
}

fn anchor(
    anchor_revision: u64,
    lineage: &[SignedAssurancePolicyRevision],
    predecessor: Option<String>,
    recorded_at_ms: u64,
) -> PolicyLineageAnchor {
    let tip = lineage.last().expect("non-empty lineage");
    PolicyLineageAnchor {
        schema_version: "1".into(),
        anchor_id: "anchor:node-1".into(),
        anchor_revision,
        manifest_id: tip.manifest.manifest_id.clone(),
        deployment_id: tip.manifest.deployment_id.clone(),
        tip_revision: tip.manifest.revision,
        tip_manifest_digest: tip.manifest.manifest_digest(),
        predecessor_anchor_digest: predecessor,
        recorded_at_ms,
        trust_store_ref: "trust-store:tpm-nv:index-1".into(),
        independent_verification_ref: format!("verification:anchor:{anchor_revision}"),
        evidence_refs: vec![format!("audit:anchor:{anchor_revision}")],
    }
}

#[allow(clippy::too_many_arguments)]
fn assess(
    case: &SafetyCase,
    current: &SignedAssurancePolicyRevision,
    lineage: &[SignedAssurancePolicyRevision],
    receipts: &[PolicyScopedSafetyReceipt],
    verifier_policy_value: &VerifierDiversityPolicy,
    signing: &SigningAuthorityGovernance,
    expected_signing_digest: &str,
    anchors: &[PolicyLineageAnchor],
    expected_anchor_digest: &str,
) -> StrictSafetyCaseStatus {
    assess_anchored_governed_readiness(
        case,
        &context(),
        &current.manifest,
        &current.signature_receipt,
        lineage,
        receipts,
        &[],
        &[],
        &[],
        &trusted_time(),
        &verifier_profiles(),
        verifier_policy_value,
        &atomic_policy(case),
        &facet_bindings(),
        signing,
        expected_signing_digest,
        anchors,
        expected_anchor_digest,
    )
    .status
}

/// Execute the reviewed governance attack/recovery matrix.
pub fn run_policy_governance_crucible() -> PolicyGovernanceCrucibleReport {
    let case = safety_case();
    let mut scenarios = Vec::new();

    // Positive control: current signed tip, authorized signer, fresh evidence, current anchor.
    let baseline_lineage = standard_lineage(&case, 2);
    let baseline_current = baseline_lineage.last().unwrap();
    let baseline_receipts = policy_receipts(
        &case,
        &baseline_current.manifest,
        &baseline_current.signature_receipt,
    );
    let baseline_signing = signing_governance(vec![]);
    let baseline_anchor = anchor(1, &baseline_lineage, None, 900);
    scenarios.push(scenario(
        "baseline-ready",
        StrictSafetyCaseStatus::Ready,
        assess(
            &case,
            baseline_current,
            &baseline_lineage,
            &baseline_receipts,
            &verifier_policy(&case, 2),
            &baseline_signing,
            &baseline_signing.governance_digest(),
            std::slice::from_ref(&baseline_anchor),
            &baseline_anchor.anchor_digest(),
        ),
    ));

    // Truncate signed history but retain external anchor at revision 2.
    let truncated = vec![baseline_lineage[0].clone()];
    let truncated_current = truncated.last().unwrap();
    let truncated_receipts = policy_receipts(
        &case,
        &truncated_current.manifest,
        &truncated_current.signature_receipt,
    );
    scenarios.push(scenario(
        "truncated-lineage-rollback-blocks",
        StrictSafetyCaseStatus::Blocked,
        assess(
            &case,
            truncated_current,
            &truncated,
            &truncated_receipts,
            &verifier_policy(&case, 2),
            &baseline_signing,
            &baseline_signing.governance_digest(),
            std::slice::from_ref(&baseline_anchor),
            &baseline_anchor.anchor_digest(),
        ),
    ));

    // Remove the middle signed revision from a three-revision lineage.
    let full_three = standard_lineage(&case, 3);
    let deleted_middle = vec![full_three[0].clone(), full_three[2].clone()];
    let deleted_current = deleted_middle.last().unwrap();
    let deleted_receipts = policy_receipts(
        &case,
        &deleted_current.manifest,
        &deleted_current.signature_receipt,
    );
    let deleted_anchor = anchor(1, &full_three, None, 900);
    scenarios.push(scenario(
        "deleted-policy-revision-invalid",
        StrictSafetyCaseStatus::Invalid,
        assess(
            &case,
            deleted_current,
            &deleted_middle,
            &deleted_receipts,
            &verifier_policy(&case, 2),
            &baseline_signing,
            &baseline_signing.governance_digest(),
            std::slice::from_ref(&deleted_anchor),
            &deleted_anchor.anchor_digest(),
        ),
    ));

    // A reviewed new manifest weakens the verifier threshold, but old receipts are
    // not automatically requalified under the new policy manifest.
    let r1 = manifest(&case, 1, None, false);
    let r2 = manifest(&case, 2, Some(r1.manifest_digest()), false);
    let r3 = manifest(&case, 3, Some(r2.manifest_digest()), true);
    let weakened_lineage = vec![
        SignedAssurancePolicyRevision {
            signature_receipt: signature(&r1, "signer:a", "key:a1"),
            manifest: r1,
        },
        SignedAssurancePolicyRevision {
            signature_receipt: signature(&r2, "signer:a", "key:a1"),
            manifest: r2.clone(),
        },
        SignedAssurancePolicyRevision {
            signature_receipt: signature(&r3, "signer:a", "key:a1"),
            manifest: r3,
        },
    ];
    let weakened_current = weakened_lineage.last().unwrap();
    let old_signature = signature(&r2, "signer:a", "key:a1");
    let old_receipts = policy_receipts(&case, &r2, &old_signature);
    let weakened_anchor = anchor(1, &weakened_lineage, None, 900);
    scenarios.push(scenario(
        "threshold-weakening-without-requalification-blocks",
        StrictSafetyCaseStatus::Blocked,
        assess(
            &case,
            weakened_current,
            &weakened_lineage,
            &old_receipts,
            &verifier_policy(&case, 1),
            &baseline_signing,
            &baseline_signing.governance_digest(),
            std::slice::from_ref(&weakened_anchor),
            &weakened_anchor.anchor_digest(),
        ),
    ));

    // Unreviewed signer/key replacement.
    let rotated = rotated_lineage(&case);
    let rotated_current = rotated.last().unwrap();
    let rotated_receipts = policy_receipts(
        &case,
        &rotated_current.manifest,
        &rotated_current.signature_receipt,
    );
    let rotated_anchor = anchor(1, &rotated, None, 900);
    scenarios.push(scenario(
        "unreviewed-signer-replacement-invalid",
        StrictSafetyCaseStatus::Invalid,
        assess(
            &case,
            rotated_current,
            &rotated,
            &rotated_receipts,
            &verifier_policy(&case, 2),
            &baseline_signing,
            &baseline_signing.governance_digest(),
            std::slice::from_ref(&rotated_anchor),
            &rotated_anchor.anchor_digest(),
        ),
    ));

    // Positive recovery: same rotation with explicit reviewed authority transition.
    let reviewed_signing = signing_governance(vec![rotation_to_b()]);
    scenarios.push(scenario(
        "reviewed-signer-rotation-ready",
        StrictSafetyCaseStatus::Ready,
        assess(
            &case,
            rotated_current,
            &rotated,
            &rotated_receipts,
            &verifier_policy(&case, 2),
            &reviewed_signing,
            &reviewed_signing.governance_digest(),
            std::slice::from_ref(&rotated_anchor),
            &rotated_anchor.anchor_digest(),
        ),
    ));

    // Substituted authority policy can authorize the replacement signer internally,
    // but cannot replace the provisioned governance digest.
    let substituted_signing = SigningAuthorityGovernance {
        policy: ManifestSigningAuthorityPolicy {
            initial_signer_ref: "signer:b".into(),
            initial_key_ref: "key:b1".into(),
            ..baseline_signing.policy.clone()
        },
        transitions: vec![],
    };
    let all_b_r1 = manifest(&case, 1, None, false);
    let all_b_r2 = manifest(&case, 2, Some(all_b_r1.manifest_digest()), false);
    let all_b_lineage = vec![
        SignedAssurancePolicyRevision {
            signature_receipt: signature(&all_b_r1, "signer:b", "key:b1"),
            manifest: all_b_r1,
        },
        SignedAssurancePolicyRevision {
            signature_receipt: signature(&all_b_r2, "signer:b", "key:b1"),
            manifest: all_b_r2,
        },
    ];
    let all_b_current = all_b_lineage.last().unwrap();
    let all_b_receipts = policy_receipts(
        &case,
        &all_b_current.manifest,
        &all_b_current.signature_receipt,
    );
    let all_b_anchor = anchor(1, &all_b_lineage, None, 900);
    scenarios.push(scenario(
        "signing-governance-substitution-invalid",
        StrictSafetyCaseStatus::Invalid,
        assess(
            &case,
            all_b_current,
            &all_b_lineage,
            &all_b_receipts,
            &verifier_policy(&case, 2),
            &substituted_signing,
            &baseline_signing.governance_digest(),
            std::slice::from_ref(&all_b_anchor),
            &all_b_anchor.anchor_digest(),
        ),
    ));

    // Old signed lineage + old anchor cannot replace a newer provisioned anchor digest.
    let old_lineage = vec![baseline_lineage[0].clone()];
    let old_current = old_lineage.last().unwrap();
    let old_anchor = anchor(1, &old_lineage, None, 800);
    let old_receipts = policy_receipts(&case, &old_current.manifest, &old_current.signature_receipt);
    scenarios.push(scenario(
        "old-anchor-substitution-invalid",
        StrictSafetyCaseStatus::Invalid,
        assess(
            &case,
            old_current,
            &old_lineage,
            &old_receipts,
            &verifier_policy(&case, 2),
            &baseline_signing,
            &baseline_signing.governance_digest(),
            std::slice::from_ref(&old_anchor),
            &baseline_anchor.anchor_digest(),
        ),
    ));

    // New signed policy revision cannot become ready before its external anchor advances.
    let forward_lineage = standard_lineage(&case, 3);
    let forward_current = forward_lineage.last().unwrap();
    let forward_receipts = policy_receipts(
        &case,
        &forward_current.manifest,
        &forward_current.signature_receipt,
    );
    scenarios.push(scenario(
        "uncheckpointed-forward-revision-blocks",
        StrictSafetyCaseStatus::Blocked,
        assess(
            &case,
            forward_current,
            &forward_lineage,
            &forward_receipts,
            &verifier_policy(&case, 2),
            &baseline_signing,
            &baseline_signing.governance_digest(),
            std::slice::from_ref(&baseline_anchor),
            &baseline_anchor.anchor_digest(),
        ),
    ));

    // Same revision number but a different manifest outside the signed lineage.
    let mut substituted_manifest = baseline_current.manifest.clone();
    substituted_manifest.policies[0].content_digest = "blake3:substituted-policy".into();
    let substituted_current = SignedAssurancePolicyRevision {
        signature_receipt: signature(&substituted_manifest, "signer:a", "key:a1"),
        manifest: substituted_manifest,
    };
    let substituted_receipts = policy_receipts(
        &case,
        &substituted_current.manifest,
        &substituted_current.signature_receipt,
    );
    scenarios.push(scenario(
        "policy-manifest-substitution-invalid",
        StrictSafetyCaseStatus::Invalid,
        assess(
            &case,
            &substituted_current,
            &baseline_lineage,
            &substituted_receipts,
            &verifier_policy(&case, 2),
            &baseline_signing,
            &baseline_signing.governance_digest(),
            std::slice::from_ref(&baseline_anchor),
            &baseline_anchor.anchor_digest(),
        ),
    ));

    // External checkpoint cannot retroactively create readiness for an earlier time.
    let late_anchor = anchor(1, &baseline_lineage, None, 1_100);
    scenarios.push(scenario(
        "late-anchor-cannot-retroactively-enable-readiness",
        StrictSafetyCaseStatus::Blocked,
        assess(
            &case,
            baseline_current,
            &baseline_lineage,
            &baseline_receipts,
            &verifier_policy(&case, 2),
            &baseline_signing,
            &baseline_signing.governance_digest(),
            std::slice::from_ref(&late_anchor),
            &late_anchor.anchor_digest(),
        ),
    ));

    // Anchor-chain deletion itself is invalid even when the final anchor points at
    // the current policy tip.
    let lineage_three = standard_lineage(&case, 3);
    let a1_lineage = vec![lineage_three[0].clone()];
    let a2_lineage = lineage_three[..2].to_vec();
    let a1 = anchor(1, &a1_lineage, None, 700);
    let a2 = anchor(2, &a2_lineage, Some(a1.anchor_digest()), 800);
    let a3 = anchor(3, &lineage_three, Some(a2.anchor_digest()), 900);
    let anchor_gap = vec![a1, a3.clone()];
    let lineage_three_current = lineage_three.last().unwrap();
    let lineage_three_receipts = policy_receipts(
        &case,
        &lineage_three_current.manifest,
        &lineage_three_current.signature_receipt,
    );
    scenarios.push(scenario(
        "deleted-anchor-revision-invalid",
        StrictSafetyCaseStatus::Invalid,
        assess(
            &case,
            lineage_three_current,
            &lineage_three,
            &lineage_three_receipts,
            &verifier_policy(&case, 2),
            &baseline_signing,
            &baseline_signing.governance_digest(),
            &anchor_gap,
            &a3.anchor_digest(),
        ),
    ));

    let status = if scenarios.iter().all(|scenario| scenario.passed) {
        PolicyGovernanceCrucibleStatus::Pass
    } else {
        PolicyGovernanceCrucibleStatus::Fail
    };
    PolicyGovernanceCrucibleReport {
        schema_version: "1".into(),
        status,
        scenarios,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn full_governance_attack_and_recovery_matrix_passes() {
        let report = run_policy_governance_crucible();
        assert_eq!(report.status, PolicyGovernanceCrucibleStatus::Pass);
        assert!(report.scenarios.iter().all(|scenario| scenario.passed));
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn required_scenarios_are_unique_and_present() {
        let report = run_policy_governance_crucible();
        let ids = report
            .scenarios
            .iter()
            .map(|scenario| scenario.scenario_id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), report.scenarios.len());
        for required in [
            "baseline-ready",
            "truncated-lineage-rollback-blocks",
            "deleted-policy-revision-invalid",
            "threshold-weakening-without-requalification-blocks",
            "unreviewed-signer-replacement-invalid",
            "reviewed-signer-rotation-ready",
            "signing-governance-substitution-invalid",
            "old-anchor-substitution-invalid",
            "uncheckpointed-forward-revision-blocks",
            "policy-manifest-substitution-invalid",
            "late-anchor-cannot-retroactively-enable-readiness",
            "deleted-anchor-revision-invalid",
        ] {
            assert!(ids.contains(required), "missing governance scenario {required}");
        }
    }
}
