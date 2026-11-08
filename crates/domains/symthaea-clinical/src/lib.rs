#![allow(dead_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![deny(clippy::dbg_macro, clippy::todo, clippy::unimplemented)]

//! # Symthaea Clinical Knowledge Ontology
//!
//! Standalone knowledge crate encoding clinical taxonomies as HDC hypervectors.
//! Covers DSM-5/ICD-11 diagnostic categories, NIMH RDoC dimensional framework,
//! therapeutic modalities (CBT/ACT/DBT/Narrative/Somatic/MI/EMDR/IFS/Psychodynamic),
//! individual symptom encoding, a conservative vocabulary for clinical claims,
//! evidence-bound clinical inference envelopes, and explicit wire identity.
//!
//! No dependency on main symthaea crate — uses only `symthaea-core` for HDC types.
//!
//! Science: APA DSM-5 (2013), WHO ICD-11 (2019), Insel et al. (2010) RDoC,
//! Barlow (2014) unified protocol, Linehan (1993) DBT, Hayes (2006) ACT.

pub mod claims;
pub mod inference;
pub mod inference_v2;
pub mod inference_wire;
pub mod inference_wire_v2;
pub mod nosology;
pub mod rdoc;
pub mod symptom_encoding;
pub mod therapeutic_modalities;

pub use claims::{
    CLINICAL_CLAIM_VOCABULARY_VERSION, ClinicalApplicability, ClinicalClaimKind,
    ClinicalClaimSemanticsV1, ClinicalClaimVocabularyError, ClinicalEvidenceStage,
    ClinicalIntendedUseClass,
};
pub use inference::{
    AlternativeClinicalHypothesisV1, CLINICAL_INFERENCE_ENVELOPE_VERSION,
    ClinicalArtifactIdentityV1, ClinicalCalibrationStatusV1, ClinicalDigestAlgorithmV1,
    ClinicalDigestV1, ClinicalDistributionAssessmentV1, ClinicalDistributionStatusV1,
    ClinicalEvidenceRefV1, ClinicalEvidenceRoleV1, ClinicalExecutionIdentityV1,
    ClinicalInferenceEnvelopeError, ClinicalInferenceEnvelopeV1, ClinicalModelIdentityV1,
    ClinicalSubjectBindingV1, ClinicalUncertaintyV1, MissingClinicalEvidenceV1,
    MissingEvidenceCriticalityV1,
};
pub use inference_v2::{
    CLINICAL_EVIDENCE_IDENTITY_V2_VERSION, CLINICAL_INFERENCE_ENVELOPE_V2_VERSION,
    ClinicalDistributionAssessmentV2, ClinicalEvidenceIdentityV2, ClinicalEvidenceRefV2,
    ClinicalExecutionIdentityV2, ClinicalInferenceEnvelopeV2, ClinicalInferenceEnvelopeV2Error,
    ClinicalModelIdentityV2, ClinicalSubjectBindingV2, ClinicalUncertaintyV2,
};
pub use inference_wire::{
    CLINICAL_INFERENCE_WIRE_IDENTITY_VERSION, ClinicalInferenceWireError,
    clinical_inference_wire_bytes, clinical_inference_wire_digest,
    clinical_inference_wire_digest_from_bytes, parse_clinical_inference_wire_bytes,
};
pub use inference_wire_v2::{
    CLINICAL_INFERENCE_WIRE_V2_VERSION, ClinicalInferenceWireDigestV2,
    ClinicalInferenceWireV2Error, clinical_inference_wire_v2_bytes,
    clinical_inference_wire_v2_digest, clinical_inference_wire_v2_digest_from_bytes,
    parse_clinical_inference_wire_v2,
};
pub use nosology::{DiagnosticCategory, DiagnosticProfile, Severity, Specifier};
pub use rdoc::{RDocDomain, RDocProfile};
pub use symptom_encoding::{SymptomEncoding, SymptomProfile};
pub use therapeutic_modalities::{
    InterventionLibrary, TherapeuticIntervention, TherapeuticModality,
};
