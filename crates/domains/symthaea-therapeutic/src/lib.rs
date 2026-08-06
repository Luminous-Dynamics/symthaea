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
//! ## ⚠ THIS CRATE DOES NOT COMPILE (as of 2026-07-29)
//!
//! Track reachability and buildability separately — this crate is **clean on
//! the first and failing on the second**, and the orphan gate passing for it
//! says nothing about its health:
//!
//! | dimension | status |
//! |---|---|
//! | module reachability | clean (no orphaned files) |
//! | compilation | **failing** |
//! | cause | `uncertainty.rs` imports `crate::model_registry::ModelExecutionReceipt`; neither the module nor the type has ever existed in any commit on any branch |
//! | since | `30b5d9ab97` ("integrate new domain patchsets"), when `uncertainty.rs` arrived |
//! | safety behaviour | fail-closed |
//! | product behaviour | model-inferred and simulated estimates unusable |
//! | resolution | explicit clinical provenance design required |
//!
//! **Do not "fix" this by deleting the field.** `EstimateEnvelope::model_receipt`
//! gates [`uncertainty::AbstentionReason::ProvenanceRequired`]: an estimate whose
//! source is `ModelInference` or `Simulation` must carry a receipt or the system
//! abstains. Because the type never existed the field can only ever be `None`, so
//! that gate currently rejects every model-inferred and simulated estimate
//! unconditionally. That is fail-closed and safe; it merely makes the feature
//! unusable. Removing the field would make the crate compile by removing a
//! clinical provenance check — trading a safe failure for an unsafe success.
//!
//! Repairing it means designing what a model-execution receipt contains on a
//! clinical safety path (model identity and immutable version/hash, execution
//! timestamp, input-classification references that do not leak sensitive
//! content, inference mode, uncertainty/calibration data, runtime identity,
//! an integrity binding, and the policy version that permitted the estimate).
//! That is a separately authorized clinical-safety task, not a cleanup item.
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
