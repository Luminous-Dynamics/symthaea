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
pub mod causal_admission;
pub mod causal_bridge;
pub mod causal_hypothesis;
pub mod causal_reasoning_bridge;
pub mod claim_evidence;
#[cfg(feature = "epistemic")]
pub mod claim_priority;
pub mod encoding;
pub mod entity_event;
pub mod epistemic_vector;
pub mod evidence_independence;
pub mod extraction;
pub mod graph;
#[cfg(feature = "epistemic")]
pub mod hdc_retrieval;
pub mod ignorance_frontier;
pub mod inquiry_contract;
pub mod llm_extraction;
pub mod manager;
pub mod persistence;
pub mod reasoning_context;
#[cfg(feature = "self_schema")]
pub mod self_schema;

pub use adaptive_ontology::{AdaptiveOntology, PrimitiveUsage};
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
pub use epistemic_vector::{
    ClaimUncertaintyAssessment, EpistemicVector, UncertaintyDimension, UncertaintyError,
    UncertaintyValue,
};
pub use evidence_independence::{
    EvidenceIndependenceAnalyzer, EvidenceLineage, ProvenanceDiversityReport, SharedAncestry,
};
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
pub use manager::{KnowledgeManager, KnowledgeSignals, KnowledgeTelemetry};
pub use reasoning_context::{
    CausalChain, EpistemicState, GroundedFact, KnowledgeQueryResult, ReasoningContext,
};
