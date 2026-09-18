// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Knowledge Engine — General-Purpose Reasoning Infrastructure
//!
//! Bridges Symthaea's HDC ontology, causal reasoning, and epistemic verification
//! into a unified knowledge system that the cognitive loop queries every cycle.
//!
//! # Architecture
//!
//! ```text
//! Input (text/percept)
//!   │
//!   ├─► Extraction ─► (entity, relation, event) tuples
//!   │                      │
//!   ├─► HDC Encoding ◄────┘   composite fact vectors
//!   │       │
//!   ├─► Knowledge Graph ◄──── temporal index, confidence decay, contradiction detection
//!   │       │
//!   ├─► Causal Bridge ◄────── auto-construct DAG edges from causal relations
//!   │       │
//!   └─► Adaptive Ontology ──► grow new primitives from experience (Hebbian)
//! ```
//!
//! # Feature Gate
//!
//! All code in this module is compiled unconditionally (no feature gate) because
//! the knowledge types are lightweight. The *cognitive loop wiring* is gated
//! behind `enable_knowledge_engine` in `CognitiveLoopConfig`.

pub mod adaptive_ontology;
#[cfg(feature = "epistemic")]
pub mod adversarial_epistemics;
mod belief_mutation_authority;
mod belief_mutation_decision_guard;
mod belief_mutation_firewall;
mod belief_mutation_persistence;
mod belief_mutation_transaction;
mod belief_mutation_verifier;
pub mod belief_revision_gate;
pub mod belief_revision_persistence;
pub mod belief_revision_receipt;
pub mod belief_revision_schema_history;
pub mod belief_revision_schema_persistence;
pub mod belief_revision_schema_wire;
pub mod belief_revision_schema_wire_validation;
pub mod belief_revision_snapshot;
pub mod causal_admission;
pub mod causal_bridge;
pub mod causal_hypothesis;
pub mod causal_reasoning_bridge;
pub mod claim_evidence;
#[cfg(feature = "epistemic")]
pub mod claim_priority;
pub mod encoding;
pub mod entity_event;
pub mod epistemic_restart_admission_review;
pub mod epistemic_restart_capsule;
pub mod epistemic_restart_capsule_v2;
pub mod epistemic_restart_continuity;
#[allow(
    dead_code,
    reason = "EKM-032 V1 reserves typed canonical-hash helpers pending canonical restore schema"
)]
pub mod epistemic_restart_manifest;
pub mod epistemic_restart_anchor;
pub mod epistemic_restart_anchor_policy;
pub mod epistemic_restart_validation_receipt;
pub mod epistemic_restart_verifier_continuity;
pub mod epistemic_restart_verifier_provenance;
#[allow(
    unused_imports,
    reason = "EKM-034 imports the support store only for its cfg(test) cold-restart fixture"
)]
pub mod epistemic_restart_wire;
pub mod epistemic_restart_wire_v2;
pub mod epistemic_restart_wire_v2_validation;
pub mod epistemic_restart_wire_validation;
pub mod epistemic_vector;
pub mod evidence_binding_conformance;
pub mod evidence_independence;
pub mod evidence_mutation_firewall;
pub mod evidence_mutation_journal;
pub mod evidence_mutation_verifier;
pub mod evidence_record_binding;
pub mod extraction;
pub mod graph;
#[cfg(feature = "epistemic")]
pub mod hdc_retrieval;
pub mod ignorance_frontier;
pub mod inquiry_contract;
pub mod inquiry_preregistration;
pub mod inquiry_result;
pub mod knowledge_weight_routing;
pub mod legacy_confidence_characterization;
pub mod llm_extraction;
pub mod manager;
pub mod persistence;
pub mod reasoning_context;
pub mod receipt_admission;
#[cfg(feature = "self_schema")]
pub mod self_schema;

pub use adaptive_ontology::{AdaptiveOntology, PrimitiveUsage};
pub use belief_mutation_authority::{
    BeliefMutationAuthority, BeliefMutationAuthorityError, PreparedBeliefMutation,
};
pub use belief_mutation_decision_guard::BeliefMutationDecisionGuardError;
#[cfg(test)]
pub(crate) use belief_mutation_firewall::BeliefMutationFirewall;
pub use belief_mutation_firewall::{
    BeliefMutationAuthorization, BeliefMutationAuthorizationDecision, BeliefMutationError,
    BeliefMutationOutcome, BeliefMutationReceipt, BeliefMutationReceiptId,
    BeliefMutationRollbackPlan, EpistemicSupportState, EpistemicSupportStore,
};
pub use belief_mutation_persistence::{
    BeliefMutationPersistenceCapsuleV1, BeliefMutationPersistenceError,
    BeliefMutationPersistenceVersion, PersistedBeliefMutationV1,
    PersistedEpistemicSupportStateV1,
};
pub use belief_mutation_transaction::{
    BeliefMutationSealError, BeliefMutationTransactionError, BeliefMutationTransactionOutcome,
    BeliefRevisionEvidenceSeal, SealedClaimSnapshot,
};
pub use belief_mutation_verifier::{
    BeliefMutationInvariantFailure, BeliefMutationSnapshot, BeliefMutationVerificationError,
    BeliefMutationVerificationReport, BeliefMutationVerifier,
};
pub use belief_revision_gate::{
    BeliefRevisionDecision, BeliefRevisionFailure, BeliefRevisionGate, BeliefRevisionPolicy,
    BeliefRevisionPolicyError, CalibrationSnapshot, EpistemicRevisionProposal,
};
pub use belief_revision_persistence::{
    BeliefRevisionHistoryCapsuleV1, BeliefRevisionPersistenceError,
    BeliefRevisionPersistenceVersion,
};
pub use belief_revision_receipt::{
    BeliefRevisionHistory, BeliefRevisionReceipt, BeliefRevisionReceiptError,
    BeliefRevisionReceiptId, RevisionEvidenceReference, RevisionEvidenceSnapshot,
};
pub use belief_revision_schema_history::{
    BeliefRevisionSchemaHistoryError, BeliefRevisionSchemaHistoryV1,
    SchemaBoundBeliefRevisionRecordV1,
};
pub use belief_revision_schema_persistence::{
    BeliefRevisionSchemaHistoryCapsuleV1, BeliefRevisionSchemaPersistenceError,
    BeliefRevisionSchemaPersistenceVersion,
};
pub use belief_revision_schema_wire::{
    BeliefRevisionSchemaWireEncoding, BeliefRevisionSchemaWireError,
    BeliefRevisionSchemaWireRecordV1, BeliefRevisionSchemaWireSnapshotV1,
    BeliefRevisionSchemaWireV1, BeliefRevisionSchemaWireVersion,
};
pub use belief_revision_schema_wire_validation::{
    BeliefRevisionSchemaWireValidationError, BeliefRevisionSchemaWireValidationReport,
    BeliefRevisionSchemaWireValidator,
};
pub use belief_revision_snapshot::{
    BeliefRevisionDecisionSnapshotV1, BeliefRevisionFailureSnapshotV1,
    BeliefRevisionPolicySchemaV1, BeliefRevisionSnapshotVersion,
    KnowledgeWeightRoutingFailureSnapshotV1, knowledge_weight_dimension_tag,
    knowledge_weight_source_tag, uncertainty_dimension_tag,
};
pub use causal_admission::{
    CausalAdmissionDecision, CausalAdmissionFailure, CausalAdmissionGate, CausalAdmissionPolicy,
};
pub use causal_bridge::CausalKnowledgeBridge;
pub use causal_hypothesis::{
    CausalEvidenceProfile, CausalHypothesis, CausalHypothesisError, CausalHypothesisId,
    CausalHypothesisStore, CausalSign,
};
pub use causal_reasoning_bridge::CausalReasoningBridge;
pub use claim_evidence::{
    ClaimId, ClaimKind, EpistemicLedger, EvidenceId, EvidenceKind, EvidencePolarity, EvidenceRecord,
    KnowledgeClaim, LedgerError, ProvenanceId, ProvenanceRecord,
};
pub use encoding::{FactEncoding, KnowledgeEncoder};
pub use entity_event::{
    EntityEventError, EntityEventStore, EntityId, EntityRelation, EventId, PersistentEntity,
    PersistentEvent, RelationId,
};
pub use epistemic_restart_capsule::{
    EpistemicRestartCapsuleError, EpistemicRestartCapsuleV1, EpistemicRestartCapsuleVersion,
    PersistedEpistemicLedgerV1, QuarantinedEpistemicRestartV1,
};
pub use epistemic_restart_capsule_v2::{
    EpistemicRestartCapsuleV2, EpistemicRestartCapsuleV2Error,
    EpistemicRestartCapsuleV2Version, EpistemicRestartV2Digest,
    QuarantinedEpistemicRestartV2,
};
pub use epistemic_restart_continuity::{
    RestartContinuityDecisionV1, RestartContinuityDispositionV1, RestartContinuityGateV1,
    TrustedRestartValidationAnchorV1,
};
pub use epistemic_restart_anchor::{
    RestartAnchorDigestV1, RestartAnchorEvidenceError, RestartAnchorEvidenceKindV1,
    RestartAnchorEvidenceV1, RestartAnchorEvidenceVerifierV1, RestartAnchorStatementV1,
    RestartAnchorTrackerV1, RestartAnchorTrackingError, VerifiedRestartAnchorEvidenceV1,
    MAX_RESTART_ANCHOR_AUTHORITY_ID_BYTES, MAX_RESTART_ANCHOR_PROOF_BYTES,
    digest_restart_anchor_statement, verify_restart_anchor_evidence,
};
pub use epistemic_restart_manifest::{
    EpistemicLedgerInventoryV1, EpistemicLedgerLineageV1, EpistemicRestartDigest,
    EpistemicRestartEncoding, EpistemicRestartManifestError, EpistemicRestartManifestV1,
    EpistemicRestartManifestVersion,
};
pub use epistemic_restart_validation_receipt::{
    EpistemicRestartValidationReceiptDigest, EpistemicRestartValidationReceiptError,
    EpistemicRestartValidationReceiptV1, EpistemicRestartValidationReceiptVersion,
    LegacyManifestAssuranceV1, RestartValidationAuthorityV1,
};
pub use epistemic_restart_wire::{
    EpistemicRestartWireEncoding, EpistemicRestartWireError, EpistemicRestartWireSnapshotV1,
    EpistemicRestartWireV1, EpistemicRestartWireVersion, WireClaimV1, WireEvidenceV1,
    WireManifestSummaryV1, WireMutationV1, WireProvenanceV1, WireRevisionBasisV1,
    WireRevisionReceiptV1, WireSupportStateV1, WireUncertaintyAssessmentV1,
};
pub use epistemic_restart_wire_v2::{
    EpistemicRestartWireSnapshotV2, EpistemicRestartWireV2, EpistemicRestartWireV2Encoding,
    EpistemicRestartWireV2Error, EpistemicRestartWireV2Version,
};
pub use epistemic_restart_wire_v2_validation::{
    EpistemicRestartWireV2ValidationError, EpistemicRestartWireV2ValidationReport,
    EpistemicRestartWireV2Validator,
};
pub use epistemic_restart_wire_validation::{
    EpistemicRestartWireValidationError, EpistemicRestartWireValidationReport,
    EpistemicRestartWireValidator,
};
pub use epistemic_vector::{
    ClaimUncertaintyAssessment, EpistemicVector, UncertaintyDimension, UncertaintyError,
    UncertaintyValue,
};
pub use evidence_binding_conformance::{
    EvidenceBindingConformanceFailure, EvidenceBindingConformanceObserver,
    EvidenceBindingConformanceReport,
};
pub use evidence_independence::{
    EvidenceIndependenceAnalyzer, EvidenceLineage, ProvenanceDiversityReport, SharedAncestry,
};
pub use evidence_mutation_firewall::{
    EvidenceDraftIdentity, EvidenceIngestionOutcome, EvidenceIngestionReceipt,
    EvidenceMutationAuthorization, EvidenceMutationError, EvidenceMutationFirewall,
    MutationAuthorizationDecision,
};
pub use evidence_mutation_journal::{
    EvidenceMutationJournal, EvidenceMutationJournalError, EvidenceMutationJournalSnapshot,
    JournaledEvidenceMutationFirewall, MutationJournalEntry,
};
pub use evidence_mutation_verifier::{
    EvidenceMutationInvariantFailure, EvidenceMutationSnapshot, EvidenceMutationVerificationError,
    EvidenceMutationVerificationReport, EvidenceMutationVerifier,
};
pub use evidence_record_binding::{EvidenceRecordBinding, EvidenceRecordBindingVersion};
pub use extraction::{
    EntityType, ExtractedEntity, ExtractedFact, ExtractedRelation, KnowledgeExtractor, SemanticRole,
};
pub use graph::{ContradictionAlert, EnhancedKnowledgeGraph, FactId, TemporalFact};
pub use ignorance_frontier::{
    ClaimIgnoranceProfile, IgnoranceFrontier, IgnoranceFrontierError, IgnoranceFrontierReport,
    KnowledgeGap,
};
pub use inquiry_contract::{
    EvidenceTarget, InquiryAuthority, InquiryContract, InquiryContractBuilder, InquiryContractError,
    InquiryContractId, InquiryPlan, InquiryRequest, ReviewRequirement,
};
pub use inquiry_preregistration::{
    DecisionInterpretation, InquiryPreregistration, PreregisteredDecisionRule,
    PreregistrationError,
};
pub use inquiry_result::{InquiryResultError, InquiryResultReceipt};
pub use knowledge_weight_routing::{
    BoundedWeight, KnowledgeWeightAuthorityRouter, KnowledgeWeightDimension, KnowledgeWeightError,
    KnowledgeWeightRoutingDecision, KnowledgeWeightRoutingFailure, KnowledgeWeightSource,
    KnowledgeWeightUpdateProposal, KnowledgeWeightVector, SignedWeightDelta,
};
pub use legacy_confidence_characterization::{
    characterize_legacy_confidence_authority, LegacyConfidenceAuthorityProfile,
};
pub use manager::{KnowledgeManager, KnowledgeSignals, KnowledgeTelemetry};
pub use reasoning_context::{
    CausalChain, EpistemicState, GroundedFact, KnowledgeQueryResult, ReasoningContext,
};
pub use receipt_admission::{
    AdmissibleEvidenceDraft, ReceiptAdmissionDecision, ReceiptAdmissionFailure,
    ReceiptAdmissionGate, ReceiptAdmissionPolicy,
};
