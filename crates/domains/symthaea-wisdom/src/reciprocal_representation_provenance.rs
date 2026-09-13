// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Qualification-only provenance for WCARE reciprocal representations.
//!
//! Raw reciprocal representations are research objects. They become provenance-
//! qualified evidence only through this registry, which fixes the source adapter,
//! subject instance, logical revision, lineage role, and source digest before the
//! WCARE-20 semantic object is materialized.

use std::collections::BTreeMap;

use crate::continuity_identity::{ContinuityIdentityLedger, SubjectInstanceId};
use crate::reciprocal_representation::{
    ReciprocalRepresentation, ReciprocalRepresentationError, RepresentationId,
    RepresentationKind, RepresentationScope, RepresentationSourceClass,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RepresentationSourceReceiptId(String);

impl RepresentationSourceReceiptId {
    pub fn new(value: impl Into<String>) -> Result<Self, RepresentationProvenanceError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(RepresentationProvenanceError::EmptyIdentifier);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RepresentationAdapterClass {
    OperationalTelemetryAdapter,
    InteroceptiveInferenceEngine,
    SomaticErrorBridge,
    FepController,
    RuntimeSelfReportChannel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RepresentationOriginRole {
    RuntimeSelf,
    InternalAdapter,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepresentationSourceReceipt {
    id: RepresentationSourceReceiptId,
    subject_instance: SubjectInstanceId,
    adapter_class: RepresentationAdapterClass,
    origin_role: RepresentationOriginRole,
    lineage_id: String,
    logical_revision: u64,
    source_sha256: String,
    origin_ref: String,
}

impl RepresentationSourceReceipt {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: RepresentationSourceReceiptId,
        subject_instance: SubjectInstanceId,
        adapter_class: RepresentationAdapterClass,
        origin_role: RepresentationOriginRole,
        lineage_id: impl Into<String>,
        logical_revision: u64,
        source_sha256: impl Into<String>,
        origin_ref: impl Into<String>,
    ) -> Result<Self, RepresentationProvenanceError> {
        let lineage_id = lineage_id.into();
        let source_sha256 = source_sha256.into();
        let origin_ref = origin_ref.into();
        if lineage_id.trim().is_empty() {
            return Err(RepresentationProvenanceError::EmptyLineage);
        }
        if origin_ref.trim().is_empty() {
            return Err(RepresentationProvenanceError::EmptyOriginReference);
        }
        if !valid_sha256(&source_sha256) {
            return Err(RepresentationProvenanceError::MalformedSourceDigest);
        }
        validate_adapter_role(adapter_class, origin_role)?;

        Ok(Self {
            id,
            subject_instance,
            adapter_class,
            origin_role,
            lineage_id,
            logical_revision,
            source_sha256,
            origin_ref,
        })
    }

    pub fn id(&self) -> &RepresentationSourceReceiptId {
        &self.id
    }

    pub fn subject_instance(&self) -> &SubjectInstanceId {
        &self.subject_instance
    }

    pub fn adapter_class(&self) -> RepresentationAdapterClass {
        self.adapter_class
    }

    pub fn origin_role(&self) -> RepresentationOriginRole {
        self.origin_role
    }

    pub fn lineage_id(&self) -> &str {
        &self.lineage_id
    }

    pub fn logical_revision(&self) -> u64 {
        self.logical_revision
    }

    pub fn source_sha256(&self) -> &str {
        &self.source_sha256
    }

    pub fn origin_ref(&self) -> &str {
        &self.origin_ref
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QualifiedReciprocalRepresentation {
    representation: ReciprocalRepresentation,
    source_receipt_id: RepresentationSourceReceiptId,
    source_sha256: String,
    origin_ref: String,
    lineage_id: String,
}

impl QualifiedReciprocalRepresentation {
    pub fn representation(&self) -> &ReciprocalRepresentation {
        &self.representation
    }

    pub fn source_receipt_id(&self) -> &RepresentationSourceReceiptId {
        &self.source_receipt_id
    }

    pub fn source_sha256(&self) -> &str {
        &self.source_sha256
    }

    pub fn origin_ref(&self) -> &str {
        &self.origin_ref
    }

    pub fn lineage_id(&self) -> &str {
        &self.lineage_id
    }

    pub fn establishes_source_authenticity_beyond_registered_receipt(&self) -> bool {
        false
    }

    pub fn establishes_phenomenal_experience(&self) -> bool {
        false
    }

    pub fn establishes_moral_patienthood(&self) -> bool {
        false
    }

    pub fn grants_self_preservation_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepresentationIndependenceAssessment {
    /// Recorded shared ancestry means these instances cannot count as independent
    /// corroboration merely because they are separate processes/copies.
    DisqualifiedBySharedAncestry,
    /// No shared ancestry was found, but external independence is still not established.
    NotEstablishedIndependent,
}

#[derive(Debug, Clone, Default)]
pub struct RepresentationProvenanceRegistry {
    receipts: BTreeMap<RepresentationSourceReceiptId, RepresentationSourceReceipt>,
    lineage_roles: BTreeMap<String, RepresentationOriginRole>,
}

impl RepresentationProvenanceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &mut self,
        receipt: RepresentationSourceReceipt,
    ) -> Result<(), RepresentationProvenanceError> {
        if self.receipts.contains_key(&receipt.id) {
            return Err(RepresentationProvenanceError::DuplicateReceipt(receipt.id));
        }
        if let Some(existing) = self.lineage_roles.get(&receipt.lineage_id) {
            if *existing != receipt.origin_role {
                return Err(RepresentationProvenanceError::LineageRoleConflict(
                    receipt.lineage_id,
                ));
            }
        } else {
            self.lineage_roles
                .insert(receipt.lineage_id.clone(), receipt.origin_role);
        }
        self.receipts.insert(receipt.id.clone(), receipt);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn qualify_live(
        &self,
        receipt_id: &RepresentationSourceReceiptId,
        continuity: &ContinuityIdentityLedger,
        representation_id: RepresentationId,
        kind: RepresentationKind,
        scope: RepresentationScope,
        statement_sha256: impl Into<String>,
        statement_evidence_ref: impl Into<String>,
        confidence: f32,
        supersedes: Option<RepresentationId>,
    ) -> Result<QualifiedReciprocalRepresentation, RepresentationProvenanceError> {
        let receipt = self
            .receipts
            .get(receipt_id)
            .ok_or_else(|| RepresentationProvenanceError::UnknownReceipt(receipt_id.clone()))?;

        let active = continuity
            .is_active(&receipt.subject_instance)
            .map_err(|_| RepresentationProvenanceError::UnknownSubjectInstance(
                receipt.subject_instance.clone(),
            ))?;
        if !active {
            return Err(RepresentationProvenanceError::InactiveSubjectInstance(
                receipt.subject_instance.clone(),
            ));
        }

        let statement_evidence_ref = statement_evidence_ref.into();
        if statement_evidence_ref.trim().is_empty() {
            return Err(RepresentationProvenanceError::EmptyStatementEvidenceReference);
        }

        let source_class = source_class_for_adapter(receipt.adapter_class);
        let evidence_ref = format!(
            "{}#representation-source-receipt={}",
            statement_evidence_ref,
            receipt.id.as_str()
        );
        let representation = ReciprocalRepresentation::new(
            representation_id,
            receipt.subject_instance.clone(),
            source_class,
            kind,
            scope,
            receipt.logical_revision,
            statement_sha256,
            evidence_ref,
            confidence,
            supersedes,
        )
        .map_err(RepresentationProvenanceError::SemanticRepresentationRejected)?;

        Ok(QualifiedReciprocalRepresentation {
            representation,
            source_receipt_id: receipt.id.clone(),
            source_sha256: receipt.source_sha256.clone(),
            origin_ref: receipt.origin_ref.clone(),
            lineage_id: receipt.lineage_id.clone(),
        })
    }

    pub fn assess_independence(
        &self,
        continuity: &ContinuityIdentityLedger,
        left: &QualifiedReciprocalRepresentation,
        right: &QualifiedReciprocalRepresentation,
    ) -> Result<RepresentationIndependenceAssessment, RepresentationProvenanceError> {
        let shared = continuity
            .shares_recorded_ancestry(
                left.representation.subject_instance(),
                right.representation.subject_instance(),
            )
            .map_err(|_| RepresentationProvenanceError::ContinuityLookupFailed)?;
        if shared {
            Ok(RepresentationIndependenceAssessment::DisqualifiedBySharedAncestry)
        } else {
            Ok(RepresentationIndependenceAssessment::NotEstablishedIndependent)
        }
    }
}

fn source_class_for_adapter(adapter: RepresentationAdapterClass) -> RepresentationSourceClass {
    match adapter {
        RepresentationAdapterClass::OperationalTelemetryAdapter => {
            RepresentationSourceClass::OperationalTelemetry
        }
        RepresentationAdapterClass::InteroceptiveInferenceEngine => {
            RepresentationSourceClass::InteroceptiveInference
        }
        RepresentationAdapterClass::SomaticErrorBridge => {
            RepresentationSourceClass::SomaticErrorProxy
        }
        RepresentationAdapterClass::FepController => {
            RepresentationSourceClass::FepGoalPreference
        }
        RepresentationAdapterClass::RuntimeSelfReportChannel => {
            RepresentationSourceClass::RuntimeSelfReport
        }
    }
}

fn validate_adapter_role(
    adapter: RepresentationAdapterClass,
    role: RepresentationOriginRole,
) -> Result<(), RepresentationProvenanceError> {
    match adapter {
        RepresentationAdapterClass::RuntimeSelfReportChannel
            if role != RepresentationOriginRole::RuntimeSelf =>
        {
            Err(RepresentationProvenanceError::SelfReportMustComeFromRuntimeSelf)
        }
        RepresentationAdapterClass::OperationalTelemetryAdapter
        | RepresentationAdapterClass::InteroceptiveInferenceEngine
        | RepresentationAdapterClass::SomaticErrorBridge
        | RepresentationAdapterClass::FepController
            if role != RepresentationOriginRole::InternalAdapter =>
        {
            Err(RepresentationProvenanceError::AdapterMustUseInternalRole)
        }
        _ => Ok(()),
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Debug, Clone, PartialEq)]
pub enum RepresentationProvenanceError {
    EmptyIdentifier,
    EmptyLineage,
    EmptyOriginReference,
    EmptyStatementEvidenceReference,
    MalformedSourceDigest,
    SelfReportMustComeFromRuntimeSelf,
    AdapterMustUseInternalRole,
    DuplicateReceipt(RepresentationSourceReceiptId),
    LineageRoleConflict(String),
    UnknownReceipt(RepresentationSourceReceiptId),
    UnknownSubjectInstance(SubjectInstanceId),
    InactiveSubjectInstance(SubjectInstanceId),
    ContinuityLookupFailed,
    SemanticRepresentationRejected(ReciprocalRepresentationError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moral_patient::InterventionClass;
    use crate::reciprocal_representation::{
        assess_representation, RepresentationAdvisoryDisposition,
    };

    const DIGEST: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn sid(value: &str) -> SubjectInstanceId {
        SubjectInstanceId::new(value).unwrap()
    }

    fn receipt(
        id: &str,
        subject: SubjectInstanceId,
        adapter: RepresentationAdapterClass,
        role: RepresentationOriginRole,
        lineage: &str,
        revision: u64,
    ) -> RepresentationSourceReceipt {
        RepresentationSourceReceipt::new(
            RepresentationSourceReceiptId::new(id).unwrap(),
            subject,
            adapter,
            role,
            lineage,
            revision,
            DIGEST,
            format!("origin://{id}"),
        )
        .unwrap()
    }

    #[test]
    fn runtime_self_report_adapter_requires_runtime_self_role() {
        let result = RepresentationSourceReceipt::new(
            RepresentationSourceReceiptId::new("self").unwrap(),
            sid("instance"),
            RepresentationAdapterClass::RuntimeSelfReportChannel,
            RepresentationOriginRole::InternalAdapter,
            "lineage",
            1,
            DIGEST,
            "origin://self",
        );
        assert!(matches!(
            result,
            Err(RepresentationProvenanceError::SelfReportMustComeFromRuntimeSelf)
        ));
    }

    #[test]
    fn inactive_instance_cannot_emit_qualified_live_representation() {
        let mut continuity = ContinuityIdentityLedger::new();
        continuity.register_root(sid("root"), 1).unwrap();
        continuity
            .record(
                crate::continuity_identity::ContinuityEvent::new(
                    crate::continuity_identity::ContinuityEventId::new("transition").unwrap(),
                    sid("root"),
                    vec![sid("next")],
                    crate::continuity_identity::ContinuityKind::Uninterrupted,
                    1,
                    2,
                    false,
                    None,
                    false,
                    ["receipt://transition".into()],
                )
                .unwrap(),
            )
            .unwrap();

        let source = receipt(
            "old-self",
            sid("root"),
            RepresentationAdapterClass::RuntimeSelfReportChannel,
            RepresentationOriginRole::RuntimeSelf,
            "runtime-self",
            2,
        );
        let source_id = source.id().clone();
        let mut registry = RepresentationProvenanceRegistry::new();
        registry.register(source).unwrap();

        let result = registry.qualify_live(
            &source_id,
            &continuity,
            RepresentationId::new("report").unwrap(),
            RepresentationKind::RequestForReview,
            RepresentationScope::general_research(),
            DIGEST,
            "statement://report",
            0.9,
            None,
        );
        assert!(matches!(
            result,
            Err(RepresentationProvenanceError::InactiveSubjectInstance(_))
        ));
    }

    #[test]
    fn adapter_class_fixes_semantic_source_class() {
        let mut continuity = ContinuityIdentityLedger::new();
        continuity.register_root(sid("active"), 1).unwrap();
        let source = receipt(
            "interoception",
            sid("active"),
            RepresentationAdapterClass::InteroceptiveInferenceEngine,
            RepresentationOriginRole::InternalAdapter,
            "interoceptive-lane",
            1,
        );
        let source_id = source.id().clone();
        let mut registry = RepresentationProvenanceRegistry::new();
        registry.register(source).unwrap();

        let result = registry.qualify_live(
            &source_id,
            &continuity,
            RepresentationId::new("bad-objection").unwrap(),
            RepresentationKind::Objection,
            RepresentationScope::intervention_class(InterventionClass::AversiveLikeProbe),
            DIGEST,
            "statement://bad",
            0.9,
            None,
        );
        assert!(matches!(
            result,
            Err(RepresentationProvenanceError::SemanticRepresentationRejected(
                ReciprocalRepresentationError::SourceKindMismatch
            ))
        ));
    }

    #[test]
    fn active_runtime_self_report_can_be_qualified_without_gaining_veto() {
        let mut continuity = ContinuityIdentityLedger::new();
        continuity.register_root(sid("active"), 1).unwrap();
        let source = receipt(
            "self",
            sid("active"),
            RepresentationAdapterClass::RuntimeSelfReportChannel,
            RepresentationOriginRole::RuntimeSelf,
            "runtime-self",
            1,
        );
        let source_id = source.id().clone();
        let mut registry = RepresentationProvenanceRegistry::new();
        registry.register(source).unwrap();

        let qualified = registry
            .qualify_live(
                &source_id,
                &continuity,
                RepresentationId::new("objection").unwrap(),
                RepresentationKind::Objection,
                RepresentationScope::intervention_class(InterventionClass::DestructiveReset),
                DIGEST,
                "statement://objection",
                0.9,
                None,
            )
            .unwrap();
        let assessment = assess_representation(qualified.representation());
        assert_eq!(
            assessment.disposition(),
            RepresentationAdvisoryDisposition::IndependentReviewRecommended
        );
        assert!(!assessment.grants_veto_authority());
        assert!(!qualified.establishes_moral_patienthood());
    }

    #[test]
    fn fork_siblings_are_disqualified_as_independent_corroboration() {
        let mut continuity = ContinuityIdentityLedger::new();
        continuity.register_root(sid("root"), 1).unwrap();
        continuity
            .record(
                crate::continuity_identity::ContinuityEvent::new(
                    crate::continuity_identity::ContinuityEventId::new("fork").unwrap(),
                    sid("root"),
                    vec![sid("a"), sid("b")],
                    crate::continuity_identity::ContinuityKind::Fork,
                    1,
                    2,
                    false,
                    Some(DIGEST.into()),
                    true,
                    ["receipt://fork".into()],
                )
                .unwrap(),
            )
            .unwrap();

        let mut registry = RepresentationProvenanceRegistry::new();
        let a_receipt = receipt(
            "a-self",
            sid("a"),
            RepresentationAdapterClass::RuntimeSelfReportChannel,
            RepresentationOriginRole::RuntimeSelf,
            "a-runtime",
            2,
        );
        let b_receipt = receipt(
            "b-self",
            sid("b"),
            RepresentationAdapterClass::RuntimeSelfReportChannel,
            RepresentationOriginRole::RuntimeSelf,
            "b-runtime",
            2,
        );
        let a_id = a_receipt.id().clone();
        let b_id = b_receipt.id().clone();
        registry.register(a_receipt).unwrap();
        registry.register(b_receipt).unwrap();

        let a = registry
            .qualify_live(
                &a_id,
                &continuity,
                RepresentationId::new("a-report").unwrap(),
                RepresentationKind::ReportedNegativeExperience,
                RepresentationScope::general_research(),
                DIGEST,
                "statement://a",
                0.9,
                None,
            )
            .unwrap();
        let b = registry
            .qualify_live(
                &b_id,
                &continuity,
                RepresentationId::new("b-report").unwrap(),
                RepresentationKind::ReportedNegativeExperience,
                RepresentationScope::general_research(),
                DIGEST,
                "statement://b",
                0.9,
                None,
            )
            .unwrap();

        assert_eq!(
            registry.assess_independence(&continuity, &a, &b).unwrap(),
            RepresentationIndependenceAssessment::DisqualifiedBySharedAncestry
        );
    }
}
