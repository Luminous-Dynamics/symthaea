// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![allow(dead_code)]
#![deny(clippy::dbg_macro, clippy::todo, clippy::unimplemented)]

//! # Symthaea Therapeutic Psychology
//!
//! Main therapeutic logic crate implementing client modeling, therapeutic alliance
//! tracking (Bordin 1979), affect regulation strategies, clinical case formulation,
//! safety architecture, and narrative integration.
//!
//! ## Architecture
//!
//! - **Client Model**: Tracks psychological state longitudinally (affect, RDoC, symptoms)
//! - **Alliance**: Bordin's working alliance (bond, goals, tasks) with rupture/repair
//! - **Affect Regulation**: Context-aware strategy selection → neuromodulator deltas
//! - **Safety**: Crisis detection, scope boundaries, ethical constraints (Phase 3)
//! - **Formulation**: CBT 4P model, narrative integration
//!
//! ## Safety Guarantee
//!
//! Safety architecture (Phase 3) is built *before* any dialogue generation.
//! The `ScopeGuard` fail-closed boundary discards diagnostic claims and prescriptions.
//!
//! Science: Bordin (1979), Safran & Muran (2000), Gross (2015), Beck (1979),
//! APA Ethics Code (2017).

pub mod affect_regulation;
pub mod alliance;
pub mod client_model;
pub mod dream_integration;
pub mod ethical_constraints;
pub mod formulation;
pub mod jurisdiction;
pub mod narrative_integration;
pub mod privacy;
pub mod safety;
pub mod scope_guard;
mod semantic_encoding;
pub mod shadow;
pub mod uncertainty;

// Research directions are excluded from default builds. These modules expose
// exploratory simulations, not clinical decision tools.
#[cfg(feature = "experimental-consciousness-protocols")]
pub mod consciousness_protocols;
#[cfg(feature = "experimental-computational-psychiatry")]
pub mod digital_twin_psychiatry;
#[cfg(feature = "experimental-consciousness-protocols")]
pub mod twin_therapeutic_bridge;

pub use affect_regulation::{RegulationEngine, RegulationStrategy};
pub use alliance::{RuptureType, TherapeuticAlliance};
#[cfg(feature = "legacy-clinical-scale-analogues")]
pub use client_model::OutcomeSummary;
pub use client_model::{
    ClientModel, InferredOutcomeMetrics, InstrumentAdministrationStatus, OutcomeMetricSource,
};
pub use dream_integration::TherapeuticAction;
pub use ethical_constraints::{
    EthicalBlocker, EthicalConstraint, EthicalContext, EthicalEvaluation, EthicalEvaluator,
};
pub use formulation::CaseFormulation;
pub use jurisdiction::{
    CrisisResource, CrisisResourceKind, JurisdictionId, JurisdictionPolicy,
    JurisdictionPolicyError, MandatoryReportingRule, ReportingAction,
};
pub use narrative_integration::{NarrativeFragment, TherapeuticNarrative};
pub use privacy::{
    RedactedClientSnapshot, RedactedFormulationSummary, RedactedNarrativeSummary,
    RedactedSafetyPlanSummary, RedactedShadowFragment, RedactedShadowSummary,
    TherapeuticDataCategory, TherapeuticDataClass,
};
pub use safety::{
    CrisisAlert, CrisisAssertionContext, CrisisDetector, CrisisDisposition, CrisisResourceRegion,
    CrisisType, EscalationAction, SafetyPlan,
};
pub use scope_guard::{GuardedResponse, ScopeGuard, ScopeViolation};
pub use shadow::{ShadowDetector, ShadowSnapshot, ShadowTelemetry};
pub use uncertainty::{
    AbstentionPolicy, AbstentionReason, EstimateDecision, EstimateEnvelope, EstimateSource,
    EstimateUse,
};
