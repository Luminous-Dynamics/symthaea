// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact deployment/configuration scope for lifecycle-managed safety evidence.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use symthaea_evidence_lifecycle::{
    EvidenceLifecycleEvent, LifecycleSafetyCaseReport, ScopedSafetyEvidenceReceipt,
    assess_lifecycle_safety_case,
};
use symthaea_formal_safety::{SafetyCase, StrictSafetyCaseStatus};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeploymentEvidenceContext {
    pub schema_version: String,
    pub deployment_id: String,
    pub configuration_digest: String,
    pub model_manifest_digest: Option<String>,
    pub calibration_manifest_digest: Option<String>,
    pub evidence_refs: Vec<String>,
}

impl DeploymentEvidenceContext {
    pub fn validate(&self) -> bool {
        !self.schema_version.trim().is_empty()
            && !self.deployment_id.trim().is_empty()
            && valid_digest(&self.configuration_digest)
            && self
                .model_manifest_digest
                .as_ref()
                .is_none_or(|value| valid_digest(value))
            && self
                .calibration_manifest_digest
                .as_ref()
                .is_none_or(|value| valid_digest(value))
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeploymentScopedSafetyReceipt {
    pub scoped_receipt: ScopedSafetyEvidenceReceipt,
    pub deployment_id: String,
    pub configuration_digest: String,
    pub model_manifest_digest: Option<String>,
    pub calibration_manifest_digest: Option<String>,
    /// Durable review record binding this receipt to the exact deployment context.
    pub scope_binding_ref: String,
}

impl DeploymentScopedSafetyReceipt {
    pub fn validate(&self) -> bool {
        self.scoped_receipt.validate()
            && !self.deployment_id.trim().is_empty()
            && valid_digest(&self.configuration_digest)
            && self
                .model_manifest_digest
                .as_ref()
                .is_none_or(|value| valid_digest(value))
            && self
                .calibration_manifest_digest
                .as_ref()
                .is_none_or(|value| valid_digest(value))
            && !self.scope_binding_ref.trim().is_empty()
    }

    pub fn matches(&self, context: &DeploymentEvidenceContext) -> bool {
        self.validate()
            && context.validate()
            && self.deployment_id == context.deployment_id
            && self.configuration_digest == context.configuration_digest
            && self.model_manifest_digest == context.model_manifest_digest
            && self.calibration_manifest_digest == context.calibration_manifest_digest
    }
}

pub fn bind_receipt_to_deployment(
    scoped_receipt: ScopedSafetyEvidenceReceipt,
    context: &DeploymentEvidenceContext,
    scope_binding_ref: impl Into<String>,
) -> Result<DeploymentScopedSafetyReceipt, DeploymentScopeError> {
    if !scoped_receipt.validate() {
        return Err(DeploymentScopeError::InvalidReceipt);
    }
    if !context.validate() {
        return Err(DeploymentScopeError::InvalidContext);
    }
    let scope_binding_ref = scope_binding_ref.into();
    if scope_binding_ref.trim().is_empty() {
        return Err(DeploymentScopeError::InvalidScopeBinding);
    }
    Ok(DeploymentScopedSafetyReceipt {
        scoped_receipt,
        deployment_id: context.deployment_id.clone(),
        configuration_digest: context.configuration_digest.clone(),
        model_manifest_digest: context.model_manifest_digest.clone(),
        calibration_manifest_digest: context.calibration_manifest_digest.clone(),
        scope_binding_ref,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeploymentScopeIssue {
    InvalidContext,
    InvalidReceipt(String),
    DuplicateReceiptId(String),
    DeploymentMismatch(String),
    ConfigurationMismatch(String),
    ModelManifestMismatch(String),
    CalibrationManifestMismatch(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeploymentScopedSafetyCaseReport {
    pub deployment_id: String,
    pub configuration_digest: String,
    pub status: StrictSafetyCaseStatus,
    pub matched_receipt_count: usize,
    pub excluded_receipt_count: usize,
    pub issues: Vec<DeploymentScopeIssue>,
    pub lifecycle_report: LifecycleSafetyCaseReport,
}

impl DeploymentScopedSafetyCaseReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeploymentScopeError {
    InvalidContext,
    InvalidReceipt,
    InvalidScopeBinding,
}

/// Assess readiness only from receipts bound to the exact current deployment
/// context. Mismatched historical receipts are excluded rather than treated as
/// malformed evidence, so configuration drift naturally removes stale support.
pub fn assess_deployment_scoped_safety_case(
    safety_case: &SafetyCase,
    context: &DeploymentEvidenceContext,
    receipts: &[DeploymentScopedSafetyReceipt],
    events: &[EvidenceLifecycleEvent],
    assessed_at_ms: u64,
) -> DeploymentScopedSafetyCaseReport {
    let mut issues = Vec::new();
    let mut matched = Vec::new();
    let mut matched_ids = BTreeSet::new();
    let mut seen_ids = BTreeSet::new();

    if !context.validate() {
        issues.push(DeploymentScopeIssue::InvalidContext);
    }

    for receipt in receipts {
        let receipt_id = receipt.scoped_receipt.receipt.receipt_id.clone();
        if !receipt.validate() {
            issues.push(DeploymentScopeIssue::InvalidReceipt(receipt_id.clone()));
            continue;
        }
        if !seen_ids.insert(receipt_id.clone()) {
            issues.push(DeploymentScopeIssue::DuplicateReceiptId(receipt_id.clone()));
            continue;
        }
        if !context.validate() {
            continue;
        }
        if receipt.deployment_id != context.deployment_id {
            issues.push(DeploymentScopeIssue::DeploymentMismatch(receipt_id));
            continue;
        }
        if receipt.configuration_digest != context.configuration_digest {
            issues.push(DeploymentScopeIssue::ConfigurationMismatch(receipt_id));
            continue;
        }
        if receipt.model_manifest_digest != context.model_manifest_digest {
            issues.push(DeploymentScopeIssue::ModelManifestMismatch(receipt_id));
            continue;
        }
        if receipt.calibration_manifest_digest != context.calibration_manifest_digest {
            issues.push(DeploymentScopeIssue::CalibrationManifestMismatch(receipt_id));
            continue;
        }
        matched_ids.insert(receipt_id);
        matched.push(receipt.scoped_receipt.clone());
    }

    // Historical events belonging to another deployment/configuration are not
    // current evidence and therefore do not enter the current lifecycle state.
    let matched_events = events
        .iter()
        .filter(|event| matched_ids.contains(&event.receipt_id))
        .cloned()
        .collect::<Vec<_>>();
    let lifecycle_report = assess_lifecycle_safety_case(
        safety_case,
        &matched,
        &matched_events,
        assessed_at_ms,
    );

    let structurally_invalid = issues.iter().any(|issue| {
        matches!(
            issue,
            DeploymentScopeIssue::InvalidContext
                | DeploymentScopeIssue::InvalidReceipt(_)
                | DeploymentScopeIssue::DuplicateReceiptId(_)
        )
    });
    let status = if structurally_invalid {
        StrictSafetyCaseStatus::Invalid
    } else {
        lifecycle_report.status
    };

    DeploymentScopedSafetyCaseReport {
        deployment_id: context.deployment_id.clone(),
        configuration_digest: context.configuration_digest.clone(),
        status,
        matched_receipt_count: matched.len(),
        excluded_receipt_count: receipts.len().saturating_sub(matched.len()),
        issues,
        lifecycle_report,
    }
}

fn valid_digest(value: &str) -> bool {
    let trimmed = value.trim();
    !trimmed.is_empty()
        && trimmed
            .split_once(':')
            .is_some_and(|(algorithm, digest)| !algorithm.is_empty() && !digest.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_evidence_lifecycle::{
        EvidenceLifecycleEventKind, ScopedSafetyEvidenceReceipt,
    };
    use symthaea_formal_safety::{
        EvidenceKind, ProofObligation, SafetyEvidenceReceipt,
    };

    fn case() -> SafetyCase {
        let mut case = SafetyCase::new("harbor-a");
        case.add_obligation(
            ProofObligation::new("claim", EvidenceKind::Test).discharge("legacy:test"),
        );
        case
    }

    fn context(config: &str) -> DeploymentEvidenceContext {
        DeploymentEvidenceContext {
            schema_version: "1".into(),
            deployment_id: "harbor-node-1".into(),
            configuration_digest: config.into(),
            model_manifest_digest: Some("blake3:model-v1".into()),
            calibration_manifest_digest: Some("blake3:cal-v1".into()),
            evidence_refs: vec!["deployment-manifest:1".into()],
        }
    }

    fn scoped(case: &SafetyCase, id: &str) -> ScopedSafetyEvidenceReceipt {
        let obligation = &case.obligations[0];
        ScopedSafetyEvidenceReceipt {
            receipt: SafetyEvidenceReceipt {
                receipt_id: id.into(),
                obligation_key: obligation.stable_key(),
                evidence_kind: EvidenceKind::Test,
                evidence_ref: format!("evidence:{id}"),
                evidence_digest: format!("blake3:{id}"),
                verifier_ref: "verifier:safety".into(),
                verified_at_ms: 100,
            },
            contract_digest: case.contract_digest(),
            valid_from_ms: 100,
            valid_until_ms: 10_000,
            applicability_refs: vec!["deployment:harbor-node-1".into()],
        }
    }

    fn bound(case: &SafetyCase, id: &str, config: &str) -> DeploymentScopedSafetyReceipt {
        bind_receipt_to_deployment(
            scoped(case, id),
            &context(config),
            format!("scope-review:{id}"),
        )
        .unwrap()
    }

    #[test]
    fn exact_context_receipt_can_support_readiness() {
        let case = case();
        let context = context("blake3:config-v1");
        let report = assess_deployment_scoped_safety_case(
            &case,
            &context,
            &[bound(&case, "r1", "blake3:config-v1")],
            &[],
            1_000,
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(report.matched_receipt_count, 1);
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn configuration_drift_removes_old_receipt_from_current_readiness() {
        let case = case();
        let old = bound(&case, "r1", "blake3:config-v1");
        let report = assess_deployment_scoped_safety_case(
            &case,
            &context("blake3:config-v2"),
            &[old],
            &[],
            1_000,
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
        assert_eq!(report.matched_receipt_count, 0);
        assert!(report
            .issues
            .contains(&DeploymentScopeIssue::ConfigurationMismatch("r1".into())));
    }

    #[test]
    fn model_manifest_drift_removes_old_evidence() {
        let case = case();
        let receipt = bound(&case, "r1", "blake3:config-v1");
        let mut current = context("blake3:config-v1");
        current.model_manifest_digest = Some("blake3:model-v2".into());
        let report = assess_deployment_scoped_safety_case(
            &case,
            &current,
            &[receipt],
            &[],
            1_000,
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
        assert!(report
            .issues
            .contains(&DeploymentScopeIssue::ModelManifestMismatch("r1".into())));
    }

    #[test]
    fn calibration_manifest_drift_removes_old_evidence() {
        let case = case();
        let receipt = bound(&case, "r1", "blake3:config-v1");
        let mut current = context("blake3:config-v1");
        current.calibration_manifest_digest = Some("blake3:cal-v2".into());
        let report = assess_deployment_scoped_safety_case(
            &case,
            &current,
            &[receipt],
            &[],
            1_000,
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
        assert!(report.issues.contains(
            &DeploymentScopeIssue::CalibrationManifestMismatch("r1".into())
        ));
    }

    #[test]
    fn new_configuration_receipt_restores_evidence_for_new_context() {
        let case = case();
        let old = bound(&case, "old", "blake3:config-v1");
        let new = bound(&case, "new", "blake3:config-v2");
        let report = assess_deployment_scoped_safety_case(
            &case,
            &context("blake3:config-v2"),
            &[old, new],
            &[],
            1_000,
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(report.matched_receipt_count, 1);
        assert_eq!(report.excluded_receipt_count, 1);
    }

    #[test]
    fn lifecycle_event_for_old_configuration_does_not_poison_new_context() {
        let case = case();
        let old = bound(&case, "old", "blake3:config-v1");
        let new = bound(&case, "new", "blake3:config-v2");
        let old_contradiction = EvidenceLifecycleEvent {
            event_id: "ev-old".into(),
            receipt_id: "old".into(),
            event_at_ms: 500,
            kind: EvidenceLifecycleEventKind::Contradicted {
                contradiction_id: "old-c1".into(),
                contradiction_ref: "incident:old-config".into(),
            },
        };
        let report = assess_deployment_scoped_safety_case(
            &case,
            &context("blake3:config-v2"),
            &[old, new],
            &[old_contradiction],
            1_000,
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(report.lifecycle_report.unresolved_contradiction_count, 0);
    }
}
