// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic metadata-only audit receipts for intimate privacy operations.
//!
//! These receipts bind privacy-operation metadata and lineage. They never store
//! intimate payload text, embeddings, preference values, or fantasy content.

use crate::intimate_memory_privacy::{
    IntimateDerivationPolicyV1, IntimateRetractionActionV1, IntimateRetractionEventV1,
    IntimateRetractionReceiptV1, IntimateRetractionStatusV1,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const INTIMATE_PRIVACY_AUDIT_SCHEMA_V1: &str =
    "symthaea.communication.intimate-privacy-audit.v1";

const RETRACTION_RECEIPT_DOMAIN_V1: &[u8] =
    b"symthaea:intimate-privacy-retraction-receipt:v1\0";
const RECOMPUTE_RECEIPT_DOMAIN_V1: &[u8] =
    b"symthaea:intimate-privacy-recompute-receipt:v1\0";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntimatePrivacyAuditReceiptV1 {
    pub operation_id: String,
    pub policy_id: String,
    pub requested_at_ns: u64,
    pub completed_at_ns: u64,
    pub source_id: String,
    pub status: IntimateRetractionStatusV1,
    pub events: Vec<IntimateRetractionEventV1>,
    pub external_deletion_unproven: bool,
    pub receipt_commitment: String,
}

impl IntimatePrivacyAuditReceiptV1 {
    pub fn from_retraction(
        operation_id: impl Into<String>,
        policy_id: impl Into<String>,
        requested_at_ns: u64,
        completed_at_ns: u64,
        retraction: &IntimateRetractionReceiptV1,
    ) -> Result<Self, IntimatePrivacyAuditErrorV1> {
        let operation_id = canonical_id(operation_id.into())
            .ok_or(IntimatePrivacyAuditErrorV1::InvalidOperationId)?;
        let policy_id = canonical_id(policy_id.into())
            .ok_or(IntimatePrivacyAuditErrorV1::InvalidPolicyId)?;
        let source_id = canonical_id(retraction.source_id.clone())
            .ok_or(IntimatePrivacyAuditErrorV1::InvalidArtifactId)?;
        if completed_at_ns < requested_at_ns {
            return Err(IntimatePrivacyAuditErrorV1::InvalidTimeOrder);
        }

        let mut events = retraction.events.clone();
        events.sort_by(|a, b| {
            a.artifact_id
                .cmp(&b.artifact_id)
                .then(a.action.cmp(&b.action))
        });

        let mut seen = BTreeSet::new();
        for event in &events {
            if canonical_id(event.artifact_id.clone()).is_none() {
                return Err(IntimatePrivacyAuditErrorV1::InvalidArtifactId);
            }
            if !seen.insert(event.artifact_id.clone()) {
                return Err(IntimatePrivacyAuditErrorV1::DuplicateArtifactEvent);
            }
        }

        match retraction.status {
            IntimateRetractionStatusV1::Applied if events.is_empty() => {
                return Err(IntimatePrivacyAuditErrorV1::AppliedReceiptMissingEvents);
            }
            IntimateRetractionStatusV1::AlreadyRetracted if !events.is_empty() => {
                return Err(IntimatePrivacyAuditErrorV1::AlreadyRetractedHasEvents);
            }
            _ => {}
        }

        let has_export_notice = events.iter().any(|event| {
            event.action == IntimateRetractionActionV1::ExternalExportNotice
        });
        if retraction.external_deletion_unproven != has_export_notice {
            return Err(IntimatePrivacyAuditErrorV1::ExternalDeletionFlagMismatch);
        }

        let mut receipt = Self {
            operation_id,
            policy_id,
            requested_at_ns,
            completed_at_ns,
            source_id,
            status: retraction.status,
            events,
            external_deletion_unproven: retraction.external_deletion_unproven,
            receipt_commitment: String::new(),
        };
        receipt.receipt_commitment = receipt.compute_commitment();
        Ok(receipt)
    }

    fn compute_commitment(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(RETRACTION_RECEIPT_DOMAIN_V1);
        hash_str(&mut hasher, INTIMATE_PRIVACY_AUDIT_SCHEMA_V1);
        hash_str(&mut hasher, &self.operation_id);
        hash_str(&mut hasher, &self.policy_id);
        hasher.update(&self.requested_at_ns.to_le_bytes());
        hasher.update(&self.completed_at_ns.to_le_bytes());
        hash_str(&mut hasher, &self.source_id);
        hasher.update(&[status_tag(self.status)]);
        hasher.update(&[u8::from(self.external_deletion_unproven)]);
        hasher.update(&(self.events.len() as u32).to_le_bytes());
        for event in &self.events {
            hash_str(&mut hasher, &event.artifact_id);
            hasher.update(&[action_tag(event.action)]);
        }
        format!("intimate-retraction-v1:{}", hasher.finalize().to_hex())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntimateRecomputeAuditReceiptV1 {
    pub operation_id: String,
    pub policy_id: String,
    pub invalidation_receipt_commitment: String,
    pub retracted_source_id: String,
    pub prior_artifact_id: String,
    pub replacement_artifact_id: String,
    pub source_ids: Vec<String>,
    pub derivation_policy: IntimateDerivationPolicyV1,
    pub completed_at_ns: u64,
    pub receipt_commitment: String,
}

impl IntimateRecomputeAuditReceiptV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        operation_id: impl Into<String>,
        policy_id: impl Into<String>,
        invalidation_receipt_commitment: impl Into<String>,
        retracted_source_id: impl Into<String>,
        prior_artifact_id: impl Into<String>,
        replacement_artifact_id: impl Into<String>,
        source_ids: Vec<String>,
        derivation_policy: IntimateDerivationPolicyV1,
        completed_at_ns: u64,
    ) -> Result<Self, IntimatePrivacyAuditErrorV1> {
        let operation_id = canonical_id(operation_id.into())
            .ok_or(IntimatePrivacyAuditErrorV1::InvalidOperationId)?;
        let policy_id = canonical_id(policy_id.into())
            .ok_or(IntimatePrivacyAuditErrorV1::InvalidPolicyId)?;
        let invalidation_receipt_commitment = canonical_id(invalidation_receipt_commitment.into())
            .ok_or(IntimatePrivacyAuditErrorV1::InvalidReceiptCommitment)?;
        let retracted_source_id = canonical_id(retracted_source_id.into())
            .ok_or(IntimatePrivacyAuditErrorV1::InvalidArtifactId)?;
        let prior_artifact_id = canonical_id(prior_artifact_id.into())
            .ok_or(IntimatePrivacyAuditErrorV1::InvalidArtifactId)?;
        let replacement_artifact_id = canonical_id(replacement_artifact_id.into())
            .ok_or(IntimatePrivacyAuditErrorV1::InvalidArtifactId)?;
        if prior_artifact_id == replacement_artifact_id {
            return Err(IntimatePrivacyAuditErrorV1::ReplacementReusesRetiredId);
        }
        if derivation_policy != IntimateDerivationPolicyV1::MustRecomputeWithoutRetractedSource {
            return Err(IntimatePrivacyAuditErrorV1::InvalidRecomputePolicy);
        }

        let mut canonical_sources = BTreeSet::new();
        for source_id in source_ids {
            let source_id = canonical_id(source_id)
                .ok_or(IntimatePrivacyAuditErrorV1::InvalidArtifactId)?;
            if source_id == retracted_source_id {
                return Err(IntimatePrivacyAuditErrorV1::RetractedSourceStillPresent);
            }
            canonical_sources.insert(source_id);
        }
        if canonical_sources.is_empty() {
            return Err(IntimatePrivacyAuditErrorV1::RecomputeMissingSources);
        }

        let mut receipt = Self {
            operation_id,
            policy_id,
            invalidation_receipt_commitment,
            retracted_source_id,
            prior_artifact_id,
            replacement_artifact_id,
            source_ids: canonical_sources.into_iter().collect(),
            derivation_policy,
            completed_at_ns,
            receipt_commitment: String::new(),
        };
        receipt.receipt_commitment = receipt.compute_commitment();
        Ok(receipt)
    }

    fn compute_commitment(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(RECOMPUTE_RECEIPT_DOMAIN_V1);
        hash_str(&mut hasher, INTIMATE_PRIVACY_AUDIT_SCHEMA_V1);
        hash_str(&mut hasher, &self.operation_id);
        hash_str(&mut hasher, &self.policy_id);
        hash_str(&mut hasher, &self.invalidation_receipt_commitment);
        hash_str(&mut hasher, &self.retracted_source_id);
        hash_str(&mut hasher, &self.prior_artifact_id);
        hash_str(&mut hasher, &self.replacement_artifact_id);
        hasher.update(&[derivation_policy_tag(self.derivation_policy)]);
        hasher.update(&self.completed_at_ns.to_le_bytes());
        hasher.update(&(self.source_ids.len() as u32).to_le_bytes());
        for source_id in &self.source_ids {
            hash_str(&mut hasher, source_id);
        }
        format!("intimate-recompute-v1:{}", hasher.finalize().to_hex())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntimatePrivacyAuditErrorV1 {
    InvalidOperationId,
    InvalidPolicyId,
    InvalidArtifactId,
    InvalidReceiptCommitment,
    InvalidTimeOrder,
    DuplicateArtifactEvent,
    AppliedReceiptMissingEvents,
    AlreadyRetractedHasEvents,
    ExternalDeletionFlagMismatch,
    ReplacementReusesRetiredId,
    InvalidRecomputePolicy,
    RetractedSourceStillPresent,
    RecomputeMissingSources,
}

fn canonical_id(value: String) -> Option<String> {
    let value = value.trim().to_owned();
    (!value.is_empty() && value.len() <= 256).then_some(value)
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    let bytes = value.as_bytes();
    hasher.update(&(bytes.len() as u32).to_le_bytes());
    hasher.update(bytes);
}

const fn status_tag(status: IntimateRetractionStatusV1) -> u8 {
    match status {
        IntimateRetractionStatusV1::Applied => 1,
        IntimateRetractionStatusV1::AlreadyRetracted => 2,
    }
}

const fn action_tag(action: IntimateRetractionActionV1) -> u8 {
    match action {
        IntimateRetractionActionV1::DeleteLocal => 1,
        IntimateRetractionActionV1::RecomputeRequired => 2,
        IntimateRetractionActionV1::RetainIndependentBasis => 3,
        IntimateRetractionActionV1::ExternalExportNotice => 4,
        IntimateRetractionActionV1::BlockUnknownDependency => 5,
    }
}

const fn derivation_policy_tag(policy: IntimateDerivationPolicyV1) -> u8 {
    match policy {
        IntimateDerivationPolicyV1::RootSource => 1,
        IntimateDerivationPolicyV1::MustDeleteOnSourceRetraction => 2,
        IntimateDerivationPolicyV1::MustRecomputeWithoutRetractedSource => 3,
        IntimateDerivationPolicyV1::MayRetainWithIndependentBasis => 4,
        IntimateDerivationPolicyV1::ExternalExportRequiresNotice => 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn retraction(events: Vec<IntimateRetractionEventV1>) -> IntimateRetractionReceiptV1 {
        IntimateRetractionReceiptV1 {
            source_id: "source".into(),
            status: IntimateRetractionStatusV1::Applied,
            external_deletion_unproven: events.iter().any(|event| {
                event.action == IntimateRetractionActionV1::ExternalExportNotice
            }),
            events,
        }
    }

    #[test]
    fn retraction_commitment_is_event_order_independent() {
        let a = retraction(vec![
            IntimateRetractionEventV1 {
                artifact_id: "summary".into(),
                action: IntimateRetractionActionV1::DeleteLocal,
            },
            IntimateRetractionEventV1 {
                artifact_id: "source".into(),
                action: IntimateRetractionActionV1::DeleteLocal,
            },
        ]);
        let b = retraction(vec![a.events[1].clone(), a.events[0].clone()]);
        let a = IntimatePrivacyAuditReceiptV1::from_retraction("op", "policy-v1", 10, 20, &a)
            .unwrap();
        let b = IntimatePrivacyAuditReceiptV1::from_retraction("op", "policy-v1", 10, 20, &b)
            .unwrap();
        assert_eq!(a.receipt_commitment, b.receipt_commitment);
        assert_eq!(a.events, b.events);
    }

    #[test]
    fn semantic_change_changes_retraction_commitment() {
        let receipt = retraction(vec![IntimateRetractionEventV1 {
            artifact_id: "source".into(),
            action: IntimateRetractionActionV1::DeleteLocal,
        }]);
        let a = IntimatePrivacyAuditReceiptV1::from_retraction("op", "policy-v1", 10, 20, &receipt)
            .unwrap();
        let b = IntimatePrivacyAuditReceiptV1::from_retraction("op", "policy-v2", 10, 20, &receipt)
            .unwrap();
        assert_ne!(a.receipt_commitment, b.receipt_commitment);
    }

    #[test]
    fn invalid_time_order_fails_closed() {
        let receipt = retraction(vec![IntimateRetractionEventV1 {
            artifact_id: "source".into(),
            action: IntimateRetractionActionV1::DeleteLocal,
        }]);
        assert_eq!(
            IntimatePrivacyAuditReceiptV1::from_retraction("op", "policy", 20, 10, &receipt),
            Err(IntimatePrivacyAuditErrorV1::InvalidTimeOrder)
        );
    }

    #[test]
    fn duplicate_artifact_events_are_rejected() {
        let receipt = retraction(vec![
            IntimateRetractionEventV1 {
                artifact_id: "source".into(),
                action: IntimateRetractionActionV1::DeleteLocal,
            },
            IntimateRetractionEventV1 {
                artifact_id: "source".into(),
                action: IntimateRetractionActionV1::RecomputeRequired,
            },
        ]);
        assert_eq!(
            IntimatePrivacyAuditReceiptV1::from_retraction("op", "policy", 10, 20, &receipt),
            Err(IntimatePrivacyAuditErrorV1::DuplicateArtifactEvent)
        );
    }

    #[test]
    fn external_deletion_limitation_is_committed() {
        let receipt = retraction(vec![
            IntimateRetractionEventV1 {
                artifact_id: "source".into(),
                action: IntimateRetractionActionV1::DeleteLocal,
            },
            IntimateRetractionEventV1 {
                artifact_id: "export".into(),
                action: IntimateRetractionActionV1::ExternalExportNotice,
            },
        ]);
        let audit = IntimatePrivacyAuditReceiptV1::from_retraction("op", "policy", 10, 20, &receipt)
            .unwrap();
        assert!(audit.external_deletion_unproven);
    }

    #[test]
    fn recompute_sources_are_canonicalized_and_order_independent() {
        let a = IntimateRecomputeAuditReceiptV1::new(
            "recompute",
            "policy",
            "intimate-retraction-v1:abc",
            "removed-source",
            "old-summary",
            "new-summary",
            vec!["source-b".into(), "source-a".into(), "source-b".into()],
            IntimateDerivationPolicyV1::MustRecomputeWithoutRetractedSource,
            30,
        )
        .unwrap();
        let b = IntimateRecomputeAuditReceiptV1::new(
            "recompute",
            "policy",
            "intimate-retraction-v1:abc",
            "removed-source",
            "old-summary",
            "new-summary",
            vec!["source-a".into(), "source-b".into()],
            IntimateDerivationPolicyV1::MustRecomputeWithoutRetractedSource,
            30,
        )
        .unwrap();
        assert_eq!(a.source_ids, vec!["source-a", "source-b"]);
        assert_eq!(a.receipt_commitment, b.receipt_commitment);
    }

    #[test]
    fn recompute_cannot_reuse_retired_identity_or_source() {
        assert_eq!(
            IntimateRecomputeAuditReceiptV1::new(
                "recompute",
                "policy",
                "receipt",
                "removed",
                "summary",
                "summary",
                vec!["source".into()],
                IntimateDerivationPolicyV1::MustRecomputeWithoutRetractedSource,
                30,
            ),
            Err(IntimatePrivacyAuditErrorV1::ReplacementReusesRetiredId)
        );
        assert_eq!(
            IntimateRecomputeAuditReceiptV1::new(
                "recompute",
                "policy",
                "receipt",
                "removed",
                "old",
                "new",
                vec!["removed".into()],
                IntimateDerivationPolicyV1::MustRecomputeWithoutRetractedSource,
                30,
            ),
            Err(IntimatePrivacyAuditErrorV1::RetractedSourceStillPresent)
        );
    }
}
