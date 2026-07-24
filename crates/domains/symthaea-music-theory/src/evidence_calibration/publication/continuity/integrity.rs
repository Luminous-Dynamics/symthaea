// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-layer integrity and authentication for continuity bundles.

use std::collections::BTreeSet;

use crate::evidence_calibration::{
    CalibrationPublicationCatalogHeadBundle,
    CalibrationPublicationCheckpointWitnessVerifier,
    CalibrationPublicationGossipConflictProof,
    CalibrationPublicationGossipLedger,
    CalibrationPublicationGossipVerifier,
    CalibrationPublicationWitnessPolicyLedger,
    CalibrationPublicationWitnessPolicyRotationVerifier,
    active_calibration_publication_witness_policy_epoch,
    audit_calibration_publication_catalog_head_bundle,
    audit_calibration_publication_gossip_conflict_proof,
    audit_calibration_publication_gossip_ledger,
    audit_calibration_publication_witness_policy_ledger,
    calibration_publication_catalog_head_bundle_sha256,
    calibration_publication_gossip_conflict_proof_sha256,
    extract_calibration_publication_gossip_conflict_proofs,
    verify_calibration_publication_catalog_head_bundle,
    verify_calibration_publication_gossip_ledger,
    verify_calibration_publication_witness_policy_ledger,
};
use crate::evidence_calibration::sha256::{Sha256, hex as sha256_hex};

use super::model::*;

pub fn build_calibration_publication_continuity_bundle<HV, RV, GV>(
    head_bundle: CalibrationPublicationCatalogHeadBundle,
    witness_policy_ledger: CalibrationPublicationWitnessPolicyLedger,
    gossip_ledger: Option<CalibrationPublicationGossipLedger>,
    conflict_proofs: Vec<CalibrationPublicationGossipConflictProof>,
    head_verifier: &HV,
    rotation_verifier: &RV,
    gossip_verifier: &GV,
) -> Result<CalibrationPublicationContinuityBundle, CalibrationPublicationContinuityError>
where
    HV: CalibrationPublicationCheckpointWitnessVerifier,
    RV: CalibrationPublicationWitnessPolicyRotationVerifier,
    GV: CalibrationPublicationGossipVerifier,
{
    let mut bundle = CalibrationPublicationContinuityBundle {
        bundle_version: CALIBRATION_PUBLICATION_CONTINUITY_BUNDLE_VERSION.into(),
        head_bundle,
        witness_policy_ledger,
        gossip_ledger,
        conflict_proofs,
        limitations: required_limitations(),
        bundle_sha256: String::new(),
    };
    bundle.bundle_sha256 = calibration_publication_continuity_bundle_sha256(&bundle);
    let report = verify_calibration_publication_continuity_bundle(
        &bundle,
        head_verifier,
        rotation_verifier,
        gossip_verifier,
    );
    if !report.authenticated() {
        return Err(CalibrationPublicationContinuityError::InvalidBundle {
            issues: report.issues.len(),
        });
    }
    Ok(bundle)
}

pub fn audit_calibration_publication_continuity_bundle(
    bundle: &CalibrationPublicationContinuityBundle,
) -> CalibrationPublicationContinuityAuditReport {
    audit_inner(
        bundle,
        None::<&NeverHeadVerifier>,
        None::<&NeverRotationVerifier>,
        None::<&NeverGossipVerifier>,
    )
}

pub fn verify_calibration_publication_continuity_bundle<HV, RV, GV>(
    bundle: &CalibrationPublicationContinuityBundle,
    head_verifier: &HV,
    rotation_verifier: &RV,
    gossip_verifier: &GV,
) -> CalibrationPublicationContinuityAuditReport
where
    HV: CalibrationPublicationCheckpointWitnessVerifier,
    RV: CalibrationPublicationWitnessPolicyRotationVerifier,
    GV: CalibrationPublicationGossipVerifier,
{
    audit_inner(
        bundle,
        Some(head_verifier),
        Some(rotation_verifier),
        Some(gossip_verifier),
    )
}

pub fn calibration_publication_continuity_bundle_sha256(
    bundle: &CalibrationPublicationContinuityBundle,
) -> String {
    let mut hash = Sha256::new();
    hash.update(CONTINUITY_BUNDLE_DOMAIN);
    hash_field(&mut hash, &bundle.bundle_version);
    hash_field(&mut hash, &bundle.head_bundle.bundle_sha256);
    hash_field(&mut hash, &bundle.witness_policy_ledger.ledger_sha256);
    hash_optional_field(
        &mut hash,
        bundle.gossip_ledger.as_ref().map(|ledger| ledger.ledger_sha256.as_str()),
    );
    let mut proofs = bundle.conflict_proofs.clone();
    proofs.sort_by(|left, right| left.proof_sha256.cmp(&right.proof_sha256));
    hash.update(&(proofs.len() as u64).to_le_bytes());
    for proof in proofs {
        hash_field(&mut hash, &proof.proof_sha256);
    }
    let mut limitations = bundle.limitations.clone();
    limitations.sort();
    hash.update(&(limitations.len() as u64).to_le_bytes());
    for limitation in limitations {
        hash.update(&[limitation_code(limitation)]);
    }
    sha256_hex(&hash.finalize())
}

fn audit_inner<HV, RV, GV>(
    bundle: &CalibrationPublicationContinuityBundle,
    head_verifier: Option<&HV>,
    rotation_verifier: Option<&RV>,
    gossip_verifier: Option<&GV>,
) -> CalibrationPublicationContinuityAuditReport
where
    HV: CalibrationPublicationCheckpointWitnessVerifier,
    RV: CalibrationPublicationWitnessPolicyRotationVerifier,
    GV: CalibrationPublicationGossipVerifier,
{
    let mut report = CalibrationPublicationContinuityAuditReport {
        audit_version: CALIBRATION_PUBLICATION_CONTINUITY_BUNDLE_AUDIT_VERSION.into(),
        structurally_valid: true,
        head_authenticated: false,
        policy_rotations_authenticated: false,
        gossip_authenticated: bundle.gossip_ledger.is_none(),
        conflict_detected: false,
        issues: Vec::new(),
    };
    if bundle.bundle_version != CALIBRATION_PUBLICATION_CONTINUITY_BUNDLE_VERSION {
        continuity_issue(&mut report, CalibrationPublicationContinuityIssueCode::BundleVersionMismatch, "publication-continuity bundle version mismatch");
    }

    let head_audit = match head_verifier {
        Some(verifier) => verify_calibration_publication_catalog_head_bundle(&bundle.head_bundle, verifier),
        None => audit_calibration_publication_catalog_head_bundle(&bundle.head_bundle),
    };
    if !head_audit.valid() {
        continuity_issue(&mut report, CalibrationPublicationContinuityIssueCode::HeadBundleInvalid, "embedded catalog-head bundle is structurally invalid");
    }
    report.head_authenticated = head_audit.accepted();
    if head_verifier.is_some() && !report.head_authenticated {
        continuity_issue(&mut report, CalibrationPublicationContinuityIssueCode::HeadWitnessAuthenticationFailed, "catalog-head witness threshold was not externally authenticated");
    }

    let policy_audit = match rotation_verifier {
        Some(verifier) => verify_calibration_publication_witness_policy_ledger(
            &bundle.witness_policy_ledger,
            verifier,
        ),
        None => audit_calibration_publication_witness_policy_ledger(&bundle.witness_policy_ledger),
    };
    if !policy_audit.valid() {
        continuity_issue(&mut report, CalibrationPublicationContinuityIssueCode::PolicyLedgerInvalid, "witness-policy history is structurally invalid");
    }
    report.policy_rotations_authenticated = policy_audit.accepted();
    if rotation_verifier.is_some() && !report.policy_rotations_authenticated {
        continuity_issue(&mut report, CalibrationPublicationContinuityIssueCode::PolicyRotationAuthenticationFailed, "one or more witness-policy rotations lack dual-quorum authentication");
    }

    let checkpoint = &bundle.head_bundle.checkpoint;
    if bundle.witness_policy_ledger.catalog_id != checkpoint.catalog_id
        || bundle.witness_policy_ledger.authority_id != checkpoint.authority_id
    {
        continuity_issue(&mut report, CalibrationPublicationContinuityIssueCode::CatalogIdentityMismatch, "head bundle and witness-policy ledger use different catalog identities");
    }
    match active_calibration_publication_witness_policy_epoch(
        &bundle.witness_policy_ledger,
        checkpoint,
    ) {
        None => continuity_issue(&mut report, CalibrationPublicationContinuityIssueCode::ActivePolicyMissing, "no witness-policy epoch is active at the packaged checkpoint"),
        Some(epoch) => {
            if epoch.policy.policy_sha256 != bundle.head_bundle.witness_set.policy.policy_sha256 {
                continuity_issue(&mut report, CalibrationPublicationContinuityIssueCode::ActivePolicyMismatch, "catalog head was witnessed under a policy other than the policy active at that height");
            }
        }
    }

    if let Some(gossip) = &bundle.gossip_ledger {
        let gossip_audit = match gossip_verifier {
            Some(verifier) => verify_calibration_publication_gossip_ledger(gossip, verifier),
            None => audit_calibration_publication_gossip_ledger(gossip),
        };
        if !gossip_audit.integrity_valid() {
            continuity_issue(&mut report, CalibrationPublicationContinuityIssueCode::GossipLedgerInvalid, "authenticated-gossip ledger is structurally invalid");
        }
        report.gossip_authenticated = gossip_audit.signatures_authenticated;
        if gossip_verifier.is_some() && !report.gossip_authenticated {
            continuity_issue(&mut report, CalibrationPublicationContinuityIssueCode::GossipAuthenticationFailed, "one or more gossip statements failed external authentication");
        }
        report.conflict_detected = !gossip_audit.conflict_free();
        if gossip.catalog_id != checkpoint.catalog_id || gossip.authority_id != checkpoint.authority_id {
            continuity_issue(&mut report, CalibrationPublicationContinuityIssueCode::GossipIdentityMismatch, "gossip ledger and catalog head use different identities");
        }
        if !gossip.statements.iter().any(|statement| {
            statement.payload.checkpoint.checkpoint_sha256 == checkpoint.checkpoint_sha256
        }) {
            continuity_issue(&mut report, CalibrationPublicationContinuityIssueCode::GossipHeadNotObserved, "gossip ledger does not contain the packaged checkpoint");
        }
        for statement in &gossip.statements {
            match active_calibration_publication_witness_policy_epoch(
                &bundle.witness_policy_ledger,
                &statement.payload.checkpoint,
            ) {
                None => continuity_issue(&mut report, CalibrationPublicationContinuityIssueCode::GossipPolicyEpochMissing, "gossip statement references a checkpoint outside the witness-policy history"),
                Some(epoch) => {
                    if epoch.epoch_sha256 != statement.payload.witness_policy_epoch_sha256 {
                        continuity_issue(&mut report, CalibrationPublicationContinuityIssueCode::GossipPolicyEpochMismatch, "gossip statement names the wrong witness-policy epoch for its checkpoint");
                    }
                }
            }
        }
        audit_conflict_proofs(bundle, gossip, &mut report);
    } else if !bundle.conflict_proofs.is_empty() {
        continuity_issue(&mut report, CalibrationPublicationContinuityIssueCode::UnexpectedConflictProof, "conflict proofs require an embedded gossip ledger");
    }

    let limitations = bundle.limitations.iter().copied().collect::<BTreeSet<_>>();
    if limitations.len() != bundle.limitations.len() {
        continuity_issue(&mut report, CalibrationPublicationContinuityIssueCode::DuplicateLimitation, "continuity limitations contain duplicates");
    }
    for required in required_limitations() {
        if !limitations.contains(&required) {
            continuity_issue(&mut report, CalibrationPublicationContinuityIssueCode::MissingLimitation, "continuity bundle omits a mandatory trust limitation");
        }
    }
    if bundle.head_bundle.bundle_sha256
        != calibration_publication_catalog_head_bundle_sha256(&bundle.head_bundle)
    {
        continuity_issue(&mut report, CalibrationPublicationContinuityIssueCode::HeadBundleInvalid, "catalog-head bundle SHA-256 mismatch");
    }
    if bundle.bundle_sha256 != calibration_publication_continuity_bundle_sha256(bundle) {
        continuity_issue(&mut report, CalibrationPublicationContinuityIssueCode::BundleSha256Mismatch, "publication-continuity bundle SHA-256 mismatch");
    }
    report.structurally_valid = report.issues.iter().all(|issue| {
        !matches!(
            issue.code,
            CalibrationPublicationContinuityIssueCode::HeadWitnessAuthenticationFailed
                | CalibrationPublicationContinuityIssueCode::PolicyRotationAuthenticationFailed
                | CalibrationPublicationContinuityIssueCode::GossipAuthenticationFailed
        )
    });
    report
}

fn audit_conflict_proofs(
    bundle: &CalibrationPublicationContinuityBundle,
    gossip: &CalibrationPublicationGossipLedger,
    report: &mut CalibrationPublicationContinuityAuditReport,
) {
    let expected = extract_calibration_publication_gossip_conflict_proofs(gossip)
        .into_iter()
        .map(|proof| proof.proof_sha256)
        .collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    for proof in &bundle.conflict_proofs {
        if !actual.insert(proof.proof_sha256.clone()) {
            continuity_issue(report, CalibrationPublicationContinuityIssueCode::DuplicateConflictProof, "continuity bundle contains a duplicate conflict proof");
        }
        if !audit_calibration_publication_gossip_conflict_proof(proof).valid()
            || proof.proof_sha256 != calibration_publication_gossip_conflict_proof_sha256(proof)
        {
            continuity_issue(report, CalibrationPublicationContinuityIssueCode::ConflictProofInvalid, "embedded gossip conflict proof is invalid");
        }
    }
    for missing in expected.difference(&actual) {
        continuity_issue(report, CalibrationPublicationContinuityIssueCode::MissingConflictProof, format!("missing conflict proof {missing}"));
    }
    for unexpected in actual.difference(&expected) {
        continuity_issue(report, CalibrationPublicationContinuityIssueCode::UnexpectedConflictProof, format!("unexpected conflict proof {unexpected}"));
    }
}

pub(crate) fn required_limitations() -> Vec<CalibrationPublicationContinuityLimitation> {
    vec![
        CalibrationPublicationContinuityLimitation::ExternalVerifiersDefineAuthentication,
        CalibrationPublicationContinuityLimitation::WitnessIndependenceNotEstablished,
        CalibrationPublicationContinuityLimitation::GossipCoverageMayBePartial,
        CalibrationPublicationContinuityLimitation::ConflictAbsenceIsNotGlobal,
        CalibrationPublicationContinuityLimitation::PolicyHistoryIsLinear,
        CalibrationPublicationContinuityLimitation::RotationQuorumsMayOverlap,
        CalibrationPublicationContinuityLimitation::ExplicitLineageIsNotCompact,
    ]
}

fn continuity_issue(
    report: &mut CalibrationPublicationContinuityAuditReport,
    code: CalibrationPublicationContinuityIssueCode,
    detail: impl Into<String>,
) {
    report.issues.push(CalibrationPublicationContinuityIssue {
        code,
        detail: detail.into(),
    });
}

fn limitation_code(value: CalibrationPublicationContinuityLimitation) -> u8 {
    match value {
        CalibrationPublicationContinuityLimitation::ExternalVerifiersDefineAuthentication => 0,
        CalibrationPublicationContinuityLimitation::WitnessIndependenceNotEstablished => 1,
        CalibrationPublicationContinuityLimitation::GossipCoverageMayBePartial => 2,
        CalibrationPublicationContinuityLimitation::ConflictAbsenceIsNotGlobal => 3,
        CalibrationPublicationContinuityLimitation::PolicyHistoryIsLinear => 4,
        CalibrationPublicationContinuityLimitation::RotationQuorumsMayOverlap => 5,
        CalibrationPublicationContinuityLimitation::ExplicitLineageIsNotCompact => 6,
    }
}

fn hash_field(hash: &mut Sha256, value: &str) {
    hash.update(&(value.len() as u64).to_le_bytes());
    hash.update(value.as_bytes());
}

fn hash_optional_field(hash: &mut Sha256, value: Option<&str>) {
    match value {
        None => hash.update(&[0]),
        Some(value) => {
            hash.update(&[1]);
            hash_field(hash, value);
        }
    }
}

struct NeverHeadVerifier;
impl CalibrationPublicationCheckpointWitnessVerifier for NeverHeadVerifier {
    type Error = &'static str;
    fn verify(
        &self,
        _payload: &[u8],
        _signer: &crate::evidence_calibration::CalibrationSignerIdentity,
        _signature: &[u8],
    ) -> Result<(), Self::Error> {
        Err("verification not attempted")
    }
}

struct NeverRotationVerifier;
impl CalibrationPublicationWitnessPolicyRotationVerifier for NeverRotationVerifier {
    type Error = &'static str;
    fn verify(
        &self,
        _payload: &[u8],
        _signer: &crate::evidence_calibration::CalibrationSignerIdentity,
        _signature: &[u8],
    ) -> Result<(), Self::Error> {
        Err("verification not attempted")
    }
}

struct NeverGossipVerifier;
impl CalibrationPublicationGossipVerifier for NeverGossipVerifier {
    type Error = &'static str;
    fn verify(
        &self,
        _payload: &[u8],
        _signer: &crate::evidence_calibration::CalibrationSignerIdentity,
        _signature: &[u8],
    ) -> Result<(), Self::Error> {
        Err("verification not attempted")
    }
}
