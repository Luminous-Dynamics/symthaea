// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-bearing scientific research primitives for Symthaea.
//!
//! This crate is intentionally dependency-light and non-executing. It defines
//! identities, bounded authority, evidence classes, independence descriptors,
//! scientific subjects, claims, polarity-aware adjudication, semantic
//! evidence-to-claim binding, protocol-scoped evidence accounting and corpus
//! decision binding, non-authorizing qualification eligibility and review-ready
//! capabilities, lifecycle-authenticated qualification-profile and claim-decision
//! authority, typed provenance graphs, frozen study protocols, immutable
//! execution/conformance records, adversarial falsification campaigns, explicit
//! scientific uncertainty budgets, and provenance-checked replication lineage
//! assessments. It does not run experiments, solvers, LLMs, HDC, statistics,
//! formal provers, or network operations.
//!
//! Ordinary records can express only `None / Declared / Bound` authority.
//! Final scientific qualification is reserved for a later capability that consumes
//! an authenticated claim decision.

pub mod authority;
pub mod claim;
pub mod claim_adjudication;
pub mod claim_binding;
pub mod evidence;
mod evidence_coverage;
pub mod evidence_coverage_binding;
pub mod evidence_coverage_gate;
pub mod execution;
pub mod falsification;
pub mod identity;
pub mod independence;
pub mod protocol;
pub mod provenance;
pub mod qualification_decision_authority;
mod qualification_eligibility;
pub mod qualification_eligibility_gate;
pub mod qualification_profile_authority;
pub mod qualification_review;
pub mod replication;
pub mod subject;
pub mod uncertainty;

pub use authority::*;
pub use claim::*;
pub use claim_adjudication::*;
pub use claim_binding::*;
pub use evidence::*;
pub use evidence_coverage::{
    CorpusDecisionKind, CorpusItemDecision, EVIDENCE_COVERAGE_SCHEMA,
    EvidenceCoverageClosure, EvidenceCoverageFinding, EvidenceCoverageIntent,
    EvidenceCoverageProtocol, EvidenceCoverageProtocolIssue, EvidenceSourceClass,
    EvidenceSourceSpec, FrozenEvidenceCoverageProtocol, RetrievalOutcome,
    SourceRetrievalReceipt,
};
pub use evidence_coverage_binding::*;
pub use evidence_coverage_gate::*;
pub use execution::*;
pub use falsification::*;
pub use identity::*;
pub use independence::*;
pub use protocol::*;
pub use provenance::*;
pub use qualification_decision_authority::*;
pub use qualification_eligibility::{
    QUALIFICATION_ELIGIBILITY_SCHEMA, EligibilityGateKind, EligibilityGateState,
    FrozenQualificationEligibilityProfile, QualificationEligibilityClosure,
    QualificationEligibilityFinding, QualificationEligibilityInputs,
    QualificationEligibilityManifest, QualificationEligibilityProfile,
    QualificationEligibilityProfileIssue,
};
pub use qualification_eligibility_gate::*;
pub use qualification_profile_authority::*;
pub use qualification_review::*;
pub use replication::*;
pub use subject::*;
pub use uncertainty::*;
