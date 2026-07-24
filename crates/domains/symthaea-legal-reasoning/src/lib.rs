// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! # symthaea-legal-reasoning
//!
//! A deterministic, dependency-free legal-reasoning microkernel: deontic
//! status, locally stratified defaults, Hohfeldian relations, explanation
//! traces, validated identifiers, and legal context.
//!
//! The crate applies already-formalized rules. It does not interpret statutes,
//! reason by analogy to precedent, infer facts from evidence, or decide what
//! the law should be.

pub mod conflict;
pub mod context;
pub mod defeasible;
pub mod deontic;
pub mod evidence;
pub mod hohfeld;
pub mod lifecycle;
pub mod model;
pub mod priority;
pub mod rules;
pub mod transition;
pub mod validation;

pub use conflict::{DefeatBasis, LegalStatus, LiteralResolution, RuleDefeat, resolve_literal};
pub use context::{
    Contextual, LegalContext, LegalDate, TemporalDimensions, TemporalError, TemporalOverlap,
    TemporalRevision, TemporalScope, governing_revisions, unique_governing_revision,
};
pub use defeasible::{
    BlockedRule, Derivation, DerivationError, DerivationStep, Rule, derive, entails,
    try_derive, try_derive_with_trace, try_entails, try_why_not,
};
pub use deontic::{
    DeonticProposition, Modality, Norm, NormAssessment, PermissionStatus, StructuredNorm,
    assess_act, assess_proposition, conflicting_acts, conflicting_propositions, is_consistent,
    is_permitted, permission_status, proposition_permission_status,
};
pub use evidence::{CanonicalEvidence, EvidenceEnvelope, EvidenceManifest};
pub use hohfeld::{Jural, JuralRelation, contradictory_relations};
pub use lifecycle::{
    ActionEvent, LifecycleAssessment, LifecycleError, NormEvent, NormState, TimedNorm,
    WaiverEvent, assess_lifecycle,
};
pub use model::{
    ActionId, Atom, AuthorityId, DocumentId, EventId, IdentifierError, JurisdictionId, Literal,
    PartyId, ProvisionId, QueryId, RevisionId, RuleId, RulePackId, SemanticProfileId, SourceRef,
};
pub use priority::{PriorityBasis, PriorityError, Superiority, SuperiorityGraph};
pub use rules::{FormalRule, RuleKind, RulePack, RulePackError};
pub use transition::{
    LegalPositionState, PowerExercise, TransitionError, TransitionRecord, exercise_power,
};
pub use validation::{Severity, ValidationIssue, ValidationReport, validate_rule_pack};
