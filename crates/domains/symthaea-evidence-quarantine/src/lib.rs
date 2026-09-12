// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-cutting quarantine for safety-evidence integrity defects that can span
//! receipts, deployments, and configurations.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use symthaea_evidence_deployment_scope::{
    DeploymentEvidenceContext, DeploymentScopedSafetyCaseReport, DeploymentScopedSafetyReceipt,
    assess_deployment_scoped_safety_case,
};
use symthaea_evidence_lifecycle::EvidenceLifecycleEvent;
use symthaea_formal_safety::{SafetyCase, StrictSafetyCaseStatus};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuarantineSelector {
    ReceiptId(String),
    EvidenceDigest(String),
    VerifierRef(String),
    EvidenceRef(String),
    ObligationKey(String),
}

impl QuarantineSelector {
    pub fn validate(&self) -> bool {
        match self {
            Self::ReceiptId(value)
            | Self::EvidenceDigest(value)
            | Self::VerifierRef(value)
            | Self::EvidenceRef(value)
            | Self::ObligationKey(value) => !value.trim().is_empty(),
        }
    }

    fn matches(&self, receipt: &DeploymentScopedSafetyReceipt) -> bool {
        let verified = &receipt.scoped_receipt.receipt;
        match self {
            Self::ReceiptId(value) => verified.receipt_id == *value,
            Self::EvidenceDigest(value) => verified.evidence_digest == *value,
            Self::VerifierRef(value) => verified.verifier_ref == *value,
            Self::EvidenceRef(value) => verified.evidence_ref == *value,
            Self::ObligationKey(value) => verified.obligation_key == *value,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceQuarantineDirective {
    pub directive_id: String,
    pub selector: QuarantineSelector,
    pub effective_from_ms: u64,
    pub reason_ref: String,
    pub evidence_refs: Vec<String>,
}

impl EvidenceQuarantineDirective {
    pub fn validate(&self) -> bool {
        !self.directive_id.trim().is_empty()
            && self.selector.validate()
            && !self.reason_ref.trim().is_empty()
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuarantineResolutionDisposition {
    /// Reviewed disposition establishes that existing evidence may be reused.
    LiftAfterReview,
    /// Existing matching receipts remain quarantined; only receipts independently
    /// verified after this resolution may be used.
    RequireReplacement,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceQuarantineResolution {
    pub resolution_id: String,
    pub directive_id: String,
    pub resolved_at_ms: u64,
    pub disposition: QuarantineResolutionDisposition,
    pub resolution_ref: String,
    pub evidence_refs: Vec<String>,
}

impl EvidenceQuarantineResolution {
    pub fn validate(&self) -> bool {
        !self.resolution_id.trim().is_empty()
            && !self.directive_id.trim().is_empty()
            && !self.resolution_ref.trim().is_empty()
            && !self.evidence_refs.is_empty()
            && self.evidence_refs.iter().all(|value| !value.trim().is_empty())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceQuarantineIssue {
    InvalidDirective(String),
    DuplicateDirectiveId(String),
    FutureDirective(String),
    InvalidResolution(String),
    DuplicateResolutionId(String),
    UnknownDirectiveResolution(String),
    FutureResolution(String),
    ResolutionPredatesDirective(String),
    MultipleResolutionsForDirective(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuarantinedReceipt {
    pub receipt_id: String,
    pub directive_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceQuarantineReport {
    pub status: StrictSafetyCaseStatus,
    pub assessed_at_ms: u64,
    pub active_directive_count: usize,
    pub quarantined_receipts: Vec<QuarantinedReceipt>,
    pub issues: Vec<EvidenceQuarantineIssue>,
    pub deployment_report: DeploymentScopedSafetyCaseReport,
}

impl EvidenceQuarantineReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

/// Apply cross-cutting quarantine before deployment/lifecycle readiness.
///
/// An unresolved directive quarantines every matching receipt. A reviewed
/// `LiftAfterReview` resolution removes that hold. `RequireReplacement` preserves
/// the hold on receipts verified on or before the resolution time and permits only
/// newly verified matching evidence afterward.
pub fn assess_quarantined_safety_case(
    safety_case: &SafetyCase,
    context: &DeploymentEvidenceContext,
    receipts: &[DeploymentScopedSafetyReceipt],
    lifecycle_events: &[EvidenceLifecycleEvent],
    directives: &[EvidenceQuarantineDirective],
    resolutions: &[EvidenceQuarantineResolution],
    assessed_at_ms: u64,
) -> EvidenceQuarantineReport {
    let mut issues = Vec::new();
    let mut directive_ids = BTreeSet::new();
    let mut directive_map = BTreeMap::<String, &EvidenceQuarantineDirective>::new();

    for directive in directives {
        if !directive.validate() {
            issues.push(EvidenceQuarantineIssue::InvalidDirective(
                directive.directive_id.clone(),
            ));
            continue;
        }
        if !directive_ids.insert(directive.directive_id.clone()) {
            issues.push(EvidenceQuarantineIssue::DuplicateDirectiveId(
                directive.directive_id.clone(),
            ));
            continue;
        }
        if directive.effective_from_ms > assessed_at_ms {
            issues.push(EvidenceQuarantineIssue::FutureDirective(
                directive.directive_id.clone(),
            ));
            continue;
        }
        directive_map.insert(directive.directive_id.clone(), directive);
    }

    let mut resolution_ids = BTreeSet::new();
    let mut resolution_by_directive = BTreeMap::<String, &EvidenceQuarantineResolution>::new();
    for resolution in resolutions {
        if !resolution.validate() {
            issues.push(EvidenceQuarantineIssue::InvalidResolution(
                resolution.resolution_id.clone(),
            ));
            continue;
        }
        if !resolution_ids.insert(resolution.resolution_id.clone()) {
            issues.push(EvidenceQuarantineIssue::DuplicateResolutionId(
                resolution.resolution_id.clone(),
            ));
            continue;
        }
        let Some(directive) = directive_map.get(&resolution.directive_id) else {
            issues.push(EvidenceQuarantineIssue::UnknownDirectiveResolution(
                resolution.directive_id.clone(),
            ));
            continue;
        };
        if resolution.resolved_at_ms > assessed_at_ms {
            issues.push(EvidenceQuarantineIssue::FutureResolution(
                resolution.resolution_id.clone(),
            ));
            continue;
        }
        if resolution.resolved_at_ms < directive.effective_from_ms {
            issues.push(EvidenceQuarantineIssue::ResolutionPredatesDirective(
                resolution.resolution_id.clone(),
            ));
            continue;
        }
        if resolution_by_directive
            .insert(resolution.directive_id.clone(), resolution)
            .is_some()
        {
            issues.push(EvidenceQuarantineIssue::MultipleResolutionsForDirective(
                resolution.directive_id.clone(),
            ));
        }
    }

    let structurally_invalid = !issues.is_empty();
    let mut retained = Vec::new();
    let mut quarantined = Vec::new();

    for receipt in receipts {
        let mut matched_directives = Vec::new();
        for (directive_id, directive) in &directive_map {
            if !directive.selector.matches(receipt) {
                continue;
            }
            let quarantines = match resolution_by_directive.get(directive_id) {
                None => true,
                Some(resolution) => match resolution.disposition {
                    QuarantineResolutionDisposition::LiftAfterReview => false,
                    QuarantineResolutionDisposition::RequireReplacement => {
                        receipt.scoped_receipt.receipt.verified_at_ms <= resolution.resolved_at_ms
                    }
                },
            };
            if quarantines {
                matched_directives.push(directive_id.clone());
            }
        }
        if matched_directives.is_empty() {
            retained.push(receipt.clone());
        } else {
            matched_directives.sort();
            quarantined.push(QuarantinedReceipt {
                receipt_id: receipt.scoped_receipt.receipt.receipt_id.clone(),
                directive_ids: matched_directives,
            });
        }
    }

    quarantined.sort_by(|a, b| a.receipt_id.cmp(&b.receipt_id));
    let retained_ids = retained
        .iter()
        .map(|receipt| receipt.scoped_receipt.receipt.receipt_id.as_str())
        .collect::<BTreeSet<_>>();
    let retained_events = lifecycle_events
        .iter()
        .filter(|event| retained_ids.contains(event.receipt_id.as_str()))
        .cloned()
        .collect::<Vec<_>>();

    let deployment_report = assess_deployment_scoped_safety_case(
        safety_case,
        context,
        &retained,
        &retained_events,
        assessed_at_ms,
    );
    let status = if structurally_invalid {
        StrictSafetyCaseStatus::Invalid
    } else {
        deployment_report.status
    };

    let active_directive_count = directive_map
        .iter()
        .filter(|(directive_id, _)| {
            !matches!(
                resolution_by_directive.get(*directive_id),
                Some(resolution)
                    if resolution.disposition == QuarantineResolutionDisposition::LiftAfterReview
            )
        })
        .count();

    EvidenceQuarantineReport {
        status,
        assessed_at_ms,
        active_directive_count,
        quarantined_receipts: quarantined,
        issues,
        deployment_report,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_evidence_deployment_scope::{
        DeploymentEvidenceContext, bind_receipt_to_deployment,
    };
    use symthaea_evidence_lifecycle::ScopedSafetyEvidenceReceipt;
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

    fn receipt(case: &SafetyCase, id: &str, verifier: &str, verified_at_ms: u64) -> DeploymentScopedSafetyReceipt {
        let obligation = &case.obligations[0];
        let scoped = ScopedSafetyEvidenceReceipt {
            receipt: SafetyEvidenceReceipt {
                receipt_id: id.into(),
                obligation_key: obligation.stable_key(),
                evidence_kind: EvidenceKind::Test,
                evidence_ref: format!("artifact:{id}"),
                evidence_digest: format!("blake3:{id}"),
                verifier_ref: verifier.into(),
                verified_at_ms,
            },
            contract_digest: case.contract_digest(),
            valid_from_ms: verified_at_ms,
            valid_until_ms: 10_000,
            applicability_refs: vec!["config:v1".into()],
        };
        bind_receipt_to_deployment(scoped, &context(), format!("scope:{id}"))
            .unwrap()
    }

    fn directive(selector: QuarantineSelector) -> EvidenceQuarantineDirective {
        EvidenceQuarantineDirective {
            directive_id: "q1".into(),
            selector,
            effective_from_ms: 500,
            reason_ref: "incident:evidence-integrity".into(),
            evidence_refs: vec!["review:q1".into()],
        }
    }

    #[test]
    fn verifier_quarantine_blocks_matching_current_evidence() {
        let case = case();
        let report = assess_quarantined_safety_case(
            &case,
            &context(),
            &[receipt(&case, "r1", "verifier:bad", 100)],
            &[],
            &[directive(QuarantineSelector::VerifierRef("verifier:bad".into()))],
            &[],
            1_000,
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
        assert_eq!(report.quarantined_receipts.len(), 1);
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn lift_after_review_can_restore_existing_receipt() {
        let case = case();
        let directive = directive(QuarantineSelector::ReceiptId("r1".into()));
        let resolution = EvidenceQuarantineResolution {
            resolution_id: "res1".into(),
            directive_id: "q1".into(),
            resolved_at_ms: 700,
            disposition: QuarantineResolutionDisposition::LiftAfterReview,
            resolution_ref: "review:false-positive-quarantine".into(),
            evidence_refs: vec!["review:res1".into()],
        };
        let report = assess_quarantined_safety_case(
            &case,
            &context(),
            &[receipt(&case, "r1", "verifier:good", 100)],
            &[],
            &[directive],
            &[resolution],
            1_000,
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Ready);
        assert!(report.quarantined_receipts.is_empty());
    }

    #[test]
    fn require_replacement_keeps_old_receipt_quarantined_but_allows_new_verification() {
        let case = case();
        let directive = directive(QuarantineSelector::VerifierRef("verifier:x".into()));
        let resolution = EvidenceQuarantineResolution {
            resolution_id: "res1".into(),
            directive_id: "q1".into(),
            resolved_at_ms: 700,
            disposition: QuarantineResolutionDisposition::RequireReplacement,
            resolution_ref: "review:verifier-remediated".into(),
            evidence_refs: vec!["review:res1".into()],
        };
        let old = receipt(&case, "old", "verifier:x", 100);
        let new = receipt(&case, "new", "verifier:x", 800);
        let report = assess_quarantined_safety_case(
            &case,
            &context(),
            &[old, new],
            &[],
            &[directive],
            &[resolution],
            1_000,
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Ready);
        assert_eq!(report.quarantined_receipts.len(), 1);
        assert_eq!(report.quarantined_receipts[0].receipt_id, "old");
        assert_eq!(report.deployment_report.matched_receipt_count, 1);
    }

    #[test]
    fn obligation_quarantine_can_span_multiple_receipts() {
        let case = case();
        let key = case.obligations[0].stable_key();
        let report = assess_quarantined_safety_case(
            &case,
            &context(),
            &[
                receipt(&case, "r1", "verifier:a", 100),
                receipt(&case, "r2", "verifier:b", 200),
            ],
            &[],
            &[directive(QuarantineSelector::ObligationKey(key))],
            &[],
            1_000,
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Blocked);
        assert_eq!(report.quarantined_receipts.len(), 2);
    }

    #[test]
    fn future_resolution_is_invalid() {
        let case = case();
        let directive = directive(QuarantineSelector::ReceiptId("r1".into()));
        let resolution = EvidenceQuarantineResolution {
            resolution_id: "future".into(),
            directive_id: "q1".into(),
            resolved_at_ms: 2_000,
            disposition: QuarantineResolutionDisposition::LiftAfterReview,
            resolution_ref: "review:future".into(),
            evidence_refs: vec!["review:future".into()],
        };
        let report = assess_quarantined_safety_case(
            &case,
            &context(),
            &[receipt(&case, "r1", "verifier:a", 100)],
            &[],
            &[directive],
            &[resolution],
            1_000,
        );
        assert_eq!(report.status, StrictSafetyCaseStatus::Invalid);
    }
}
