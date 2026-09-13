// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed Engineering Trust Kernel boundary for native analytical evidence.
//!
//! ```text
//! native calculation
//! != admitted analytical evidence
//! != historical discharge receipt
//! != bounded current analytical discharge fact
//! != requirement satisfaction
//! != qualified design / certification / manufacturing / deployment / actuation
//! ```
//!
//! Native analytical evidence remains distinct from external-solver `Simulation`
//! evidence. Authority-bearing IDs are one-way capabilities; semantic context
//! identities are derived from explicit records rather than caller-supplied
//! hash-shaped strings. External content premises are role-safe typed digests.
//!
//! Present-tense analytical authority is bounded: a currentness assertion has an
//! explicit observation and expiry time, and current-fact derivation requires an
//! evaluation time inside that inclusive interval.

#![deny(unsafe_code)]

mod analysis;
mod authority;
mod canonical;
mod context;

pub use analysis::{
    AnalyticalAcceptancePolicyV1, AnalyticalMethodV1, NativeAnalyticalResultV1,
    RectangularCantileverInputV1,
};
pub use authority::{
    AdmittedAnalyticalEvidenceV1, CurrentNativeAnalyticalDischargeFactV2,
    NativeAnalyticalDischargeReceiptV1, NativeAnalyticalPlanV1,
    admit_native_analytical_evidence_v1, derive_current_native_analytical_discharge_fact_v2,
    issue_native_analytical_discharge_receipt_v1,
};
pub use canonical::{
    AcceptanceRecordDigestV1, AdmittedAnalyticalEvidenceIdV1, AlgorithmRevisionDigestV1,
    AnalysisConfigurationDigestV1, AnalysisRequirementRevisionIdV1, AnalysisTrustErrorV1,
    AnalyticalInputRevisionIdV1, AnalyticalMethodRevisionIdV1, AnalyticalPlanIdV1,
    AnalyticalPolicyRevisionIdV1, CurrentNativeAnalyticalDischargeFactIdV2,
    CurrentnessAssertionIdV2, CurrentnessAttestationDigestV1, ExecutionArtifactDigestV1,
    ImplementationArtifactDigestV1, ModelQualificationRecordDigestV1, ModelRevisionDigestV1,
    NativeAnalyticalDischargeReceiptIdV1, ObligationRevisionIdV1, Sha256DigestV1,
    SubjectRevisionIdV1, SubjectStateDigestV1, TwinRevisionIdV1, TwinSchemaDigestV1,
    TwinStateDigestV1, ValidityDimensionDigestV1, ValidityDomainRevisionIdV1,
    canonical_binary64_v1,
};
pub use context::{
    AcceptedAnalysisRequirementV1, CurrentnessAssertionV2, SubjectRevisionV1, TwinKindV1,
    TwinRevisionV1, ValidityDomainRevisionV1, analytical_obligation_revision_v1,
};
