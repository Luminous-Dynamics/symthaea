// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Replay contracts for release quorum, timed sessions, audit anchors, and
//! deterministic fault-containment evidence.

use crate::audit_evidence::{VerifiedAuditAnchor, digest_audit_anchor};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::fault_injection::{FaultInjectionReport, FaultScenario, verify_fault_injection_report};
use crate::release::ReleaseAuthority;
use crate::replay::GovernedFabricationReplayContract;
use crate::submission::GovernedSubmittedJobReceipt;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const OPERATIONAL_REPLAY_SCHEMA: &str = "symthaea.fabrication.operational-replay-contract.v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationalFabricationReplayContract {
    pub schema_version: String,
    pub governed: GovernedFabricationReplayContract,
    pub release_policy_digest: Sha256Digest,
    pub delegation_digest: Option<Sha256Digest>,
    pub session_digest: Sha256Digest,
    pub session_sequence: u64,
    pub audit_anchor_digest: Sha256Digest,
    pub fault_matrix_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationalReplayError {
    ManifestMismatch,
    TrustSnapshotMismatch,
    ReleasePolicyMismatch,
    DelegationMismatch,
    SessionMismatch,
    AuditAnchorMismatch,
    EmptyFaultMatrix,
    DuplicateFaultScenario(FaultScenario),
    IncompleteFaultMatrix,
    InvalidFaultReport(FaultScenario),
    AuditAnchorEncoding(String),
    ContractEncoding(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationalReplayMismatch {
    SchemaVersion,
    GovernedContract,
    ReleasePolicy,
    Delegation,
    SessionDigest,
    SessionSequence,
    AuditAnchor,
    FaultMatrix,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationalReplayVerificationReport {
    pub mismatches: Vec<OperationalReplayMismatch>,
}

impl OperationalReplayVerificationReport {
    pub fn reproducible(&self) -> bool {
        self.mismatches.is_empty()
    }
}

pub fn build_operational_replay_contract(
    governed: GovernedFabricationReplayContract,
    release: &dyn ReleaseAuthority,
    receipt: &GovernedSubmittedJobReceipt,
    audit_anchor: &VerifiedAuditAnchor,
    fault_reports: &[FaultInjectionReport],
) -> Result<OperationalFabricationReplayContract, OperationalReplayError> {
    if governed.base.manifest_digest != release.manifest_digest()
        || governed.base.manifest_digest != receipt.submission.manifest_digest
    {
        return Err(OperationalReplayError::ManifestMismatch);
    }
    if governed.trust_snapshot_digest != release.trust_snapshot_digest()
        || governed.trust_snapshot_digest != receipt.trust_snapshot_digest
    {
        return Err(OperationalReplayError::TrustSnapshotMismatch);
    }
    if receipt.release_policy_digest != release.policy_digest() {
        return Err(OperationalReplayError::ReleasePolicyMismatch);
    }
    if receipt.delegation_digest != release.delegation_digest() {
        return Err(OperationalReplayError::DelegationMismatch);
    }
    if receipt.session_sequence == 0 {
        return Err(OperationalReplayError::SessionMismatch);
    }
    if audit_anchor.anchor().journal_digest != governed.audit_journal_digest
        || audit_anchor.anchor().journal_head != governed.audit_head
        || audit_anchor.anchor().trust_snapshot_digest != governed.trust_snapshot_digest
    {
        return Err(OperationalReplayError::AuditAnchorMismatch);
    }
    let audit_anchor_digest = digest_audit_anchor(audit_anchor.anchor())
        .map_err(|error| OperationalReplayError::AuditAnchorEncoding(format!("{error:?}")))?;
    let fault_matrix_digest = digest_fault_injection_matrix(fault_reports)?;
    Ok(OperationalFabricationReplayContract {
        schema_version: OPERATIONAL_REPLAY_SCHEMA.into(),
        governed,
        release_policy_digest: release.policy_digest(),
        delegation_digest: release.delegation_digest(),
        session_digest: receipt.session_digest,
        session_sequence: receipt.session_sequence,
        audit_anchor_digest,
        fault_matrix_digest,
    })
}

pub fn digest_operational_replay_contract(
    contract: &OperationalFabricationReplayContract,
) -> Result<Sha256Digest, OperationalReplayError> {
    let bytes = serde_json::to_vec(contract)
        .map_err(|error| OperationalReplayError::ContractEncoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.operational-replay-contract-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn verify_operational_replay_contract(
    contract: &OperationalFabricationReplayContract,
    governed: GovernedFabricationReplayContract,
    release: &dyn ReleaseAuthority,
    receipt: &GovernedSubmittedJobReceipt,
    audit_anchor: &VerifiedAuditAnchor,
    fault_reports: &[FaultInjectionReport],
) -> Result<OperationalReplayVerificationReport, OperationalReplayError> {
    let expected =
        build_operational_replay_contract(governed, release, receipt, audit_anchor, fault_reports)?;
    let mut mismatches = Vec::new();
    if contract.schema_version != OPERATIONAL_REPLAY_SCHEMA {
        mismatches.push(OperationalReplayMismatch::SchemaVersion);
    }
    if contract.governed != expected.governed {
        mismatches.push(OperationalReplayMismatch::GovernedContract);
    }
    if contract.release_policy_digest != expected.release_policy_digest {
        mismatches.push(OperationalReplayMismatch::ReleasePolicy);
    }
    if contract.delegation_digest != expected.delegation_digest {
        mismatches.push(OperationalReplayMismatch::Delegation);
    }
    if contract.session_digest != expected.session_digest {
        mismatches.push(OperationalReplayMismatch::SessionDigest);
    }
    if contract.session_sequence != expected.session_sequence {
        mismatches.push(OperationalReplayMismatch::SessionSequence);
    }
    if contract.audit_anchor_digest != expected.audit_anchor_digest {
        mismatches.push(OperationalReplayMismatch::AuditAnchor);
    }
    if contract.fault_matrix_digest != expected.fault_matrix_digest {
        mismatches.push(OperationalReplayMismatch::FaultMatrix);
    }
    Ok(OperationalReplayVerificationReport { mismatches })
}

pub fn digest_fault_injection_matrix(
    reports: &[FaultInjectionReport],
) -> Result<Sha256Digest, OperationalReplayError> {
    if reports.is_empty() {
        return Err(OperationalReplayError::EmptyFaultMatrix);
    }
    let mut inventory = Vec::with_capacity(reports.len());
    let mut scenarios = BTreeSet::new();
    for report in reports {
        if !verify_fault_injection_report(report)
            .map_err(|_| OperationalReplayError::InvalidFaultReport(report.scenario))?
        {
            return Err(OperationalReplayError::InvalidFaultReport(report.scenario));
        }
        if !scenarios.insert(report.scenario) {
            return Err(OperationalReplayError::DuplicateFaultScenario(
                report.scenario,
            ));
        }
        inventory.push((report.scenario, report.report_digest));
    }
    let required = BTreeSet::from([
        FaultScenario::HeartbeatLoss,
        FaultScenario::ProgressStall,
        FaultScenario::NozzleRunaway,
        FaultScenario::BedRunaway,
        FaultScenario::NozzleControlDeviation,
        FaultScenario::BedControlDeviation,
        FaultScenario::TimeRegression,
        FaultScenario::ProgressRegression,
        FaultScenario::NonFiniteSensor,
    ]);
    if scenarios != required {
        return Err(OperationalReplayError::IncompleteFaultMatrix);
    }
    inventory.sort_by_key(|(scenario, _)| *scenario);
    let bytes = serde_json::to_vec(&inventory)
        .map_err(|error| OperationalReplayError::ContractEncoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.fault-matrix-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_guard::ExecutionGuardPolicy;
    use crate::fault_injection::run_standard_fault_matrix;

    #[test]
    fn fault_matrix_digest_is_order_independent_and_complete() {
        let reports = run_standard_fault_matrix(ExecutionGuardPolicy::default()).unwrap();
        let left = digest_fault_injection_matrix(&reports).unwrap();
        let mut reversed = reports.clone();
        reversed.reverse();
        assert_eq!(left, digest_fault_injection_matrix(&reversed).unwrap());
        reversed.pop();
        assert!(matches!(
            digest_fault_injection_matrix(&reversed),
            Err(OperationalReplayError::IncompleteFaultMatrix)
        ));
    }
}
