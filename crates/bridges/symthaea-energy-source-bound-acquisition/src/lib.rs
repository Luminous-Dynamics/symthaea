// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Source-identity-bound prospective acquisition for Tier-1 energy-material campaigns.
//!
//! The v0 campaign declaration binds the acquisition query, acquired artifact and
//! source receipt, but it cannot express the actual acquired source name/version/URI.
//! This additive v1 wrapper closes that gap without silently changing old receipt
//! semantics. It validates exact source identity against the frozen campaign and
//! then delegates the existing compatibility/native admission engines.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use symthaea_energy_material_campaign::{
    admit_campaign_result, AcquisitionDeclaration, CampaignAdmissionReceipt, CampaignError,
    SourceCommitment, Tier1CampaignManifest,
    CAPABILITY_CLASSIFICATION as COMPATIBILITY_CAPABILITY_CLASSIFICATION,
};
use symthaea_energy_material_candidate_version::CandidateVersionBoundDossier;
use symthaea_energy_material_screening::EvidenceDimension;
use symthaea_energy_native_campaign_admission::{
    admit_native_campaign_result, NativeAdmissionError, NativeCampaignAdmissionReceipt,
    CAPABILITY_CLASSIFICATION as NATIVE_CAPABILITY_CLASSIFICATION,
};
use symthaea_energy_native_dossier::NativeEnvelopeDossier;
use thiserror::Error;

pub const CAPABILITY_CLASSIFICATION: &str =
    "SOURCE-BOUND PROSPECTIVE ACQUISITION ADMISSION ONLY -- exact source identity and evidence-plan conformance are not source authority, reviewer authentication, scientific validation, candidate promotion, certification, or deployment authority.";

const COMPAT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.energy-material.source-bound-compatibility-admission.v1\0";
const NATIVE_DIGEST_DOMAIN: &[u8] =
    b"symthaea.energy-material.source-bound-native-admission.v1\0";

/// Actual source identity asserted for a prospective acquisition.
///
/// Equality is intentionally exact. URI/version normalization belongs in an
/// explicit upstream acquisition policy; admission must not reinterpret a
/// preregistered identity after results exist.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcquisitionSourceIdentity {
    pub source_name: String,
    pub source_version: String,
    pub source_uri: String,
}

impl AcquisitionSourceIdentity {
    pub fn validate(&self) -> Result<(), SourceBoundAdmissionError> {
        if self.source_name.trim().is_empty()
            || self.source_version.trim().is_empty()
            || self.source_uri.trim().is_empty()
        {
            return Err(SourceBoundAdmissionError::InvalidDeclaration(
                "source name/version/URI must be non-empty".into(),
            ));
        }
        Ok(())
    }
}

/// v1 prospective acquisition declaration that preserves actual source identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceBoundAcquisitionDeclaration {
    pub dimension: EvidenceDimension,
    pub source: AcquisitionSourceIdentity,
    pub acquisition_query_sha256: String,
    pub acquired_artifact_sha256: String,
    pub source_receipt_sha256: String,
    pub reviewer: String,
    pub note: String,
}

impl SourceBoundAcquisitionDeclaration {
    pub fn validate(&self) -> Result<(), SourceBoundAdmissionError> {
        self.source.validate()?;
        for (name, digest) in [
            ("acquisition query", self.acquisition_query_sha256.as_str()),
            ("acquired artifact", self.acquired_artifact_sha256.as_str()),
            ("source receipt", self.source_receipt_sha256.as_str()),
        ] {
            validate_sha256(digest, name)?;
        }
        if self.reviewer.trim().is_empty() || self.note.trim().is_empty() {
            return Err(SourceBoundAdmissionError::InvalidDeclaration(
                "reviewer and note must be non-empty".into(),
            ));
        }
        Ok(())
    }

    fn to_v0(&self) -> AcquisitionDeclaration {
        AcquisitionDeclaration {
            dimension: self.dimension,
            acquisition_query_sha256: self.acquisition_query_sha256.clone(),
            acquired_artifact_sha256: self.acquired_artifact_sha256.clone(),
            source_receipt_sha256: self.source_receipt_sha256.clone(),
            reviewer: self.reviewer.clone(),
            note: self.note.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceBoundCompatibilityAdmissionReceipt {
    pub schema: String,
    pub capability_classification: String,
    pub campaign_manifest_sha256: String,
    pub acquisition_declarations: Vec<SourceBoundAcquisitionDeclaration>,
    pub compatibility_admission_sha256: String,
    pub compatibility_admission: CampaignAdmissionReceipt,
}

impl SourceBoundCompatibilityAdmissionReceipt {
    pub fn validate(&self) -> Result<(), SourceBoundAdmissionError> {
        if self.schema != "symthaea.energy-material.source-bound-compatibility-admission.v1"
            || self.capability_classification != CAPABILITY_CLASSIFICATION
        {
            return Err(SourceBoundAdmissionError::InvalidReceipt(
                "receipt schema/capability classification was altered".into(),
            ));
        }
        validate_sha256(&self.campaign_manifest_sha256, "campaign manifest")?;
        validate_sha256(
            &self.compatibility_admission_sha256,
            "compatibility admission",
        )?;
        validate_declaration_structure(&self.acquisition_declarations)?;
        validate_compatibility_inner(&self.compatibility_admission)?;
        if self.compatibility_admission.campaign_manifest_sha256
            != self.campaign_manifest_sha256
        {
            return Err(SourceBoundAdmissionError::InvalidReceipt(
                "inner compatibility admission belongs to a different campaign".into(),
            ));
        }
        let expected_v0 = v0_declarations(&self.acquisition_declarations);
        if self.compatibility_admission.acquisition_declarations != expected_v0 {
            return Err(SourceBoundAdmissionError::InnerDeclarationMismatch);
        }
        let expected = self.compatibility_admission.sha256()?;
        if self.compatibility_admission_sha256 != expected {
            return Err(SourceBoundAdmissionError::InnerAdmissionDigestMismatch {
                expected,
                actual: self.compatibility_admission_sha256.clone(),
            });
        }
        Ok(())
    }

    pub fn validate_with_inputs(
        &self,
        manifest: &Tier1CampaignManifest,
        dossier: &CandidateVersionBoundDossier,
    ) -> Result<(), SourceBoundAdmissionError> {
        self.validate()?;
        let recomputed = admit_source_bound_compatibility(
            manifest,
            dossier,
            self.acquisition_declarations.clone(),
        )?;
        if recomputed != *self {
            return Err(SourceBoundAdmissionError::ReceiptReplayMismatch);
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, SourceBoundAdmissionError> {
        self.validate()?;
        domain_hash(COMPAT_DIGEST_DOMAIN, self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceBoundNativeAdmissionReceipt {
    pub schema: String,
    pub capability_classification: String,
    pub campaign_manifest_sha256: String,
    pub acquisition_declarations: Vec<SourceBoundAcquisitionDeclaration>,
    pub native_admission_sha256: String,
    pub native_admission: NativeCampaignAdmissionReceipt,
}

impl SourceBoundNativeAdmissionReceipt {
    pub fn validate(&self) -> Result<(), SourceBoundAdmissionError> {
        if self.schema != "symthaea.energy-material.source-bound-native-admission.v1"
            || self.capability_classification != CAPABILITY_CLASSIFICATION
        {
            return Err(SourceBoundAdmissionError::InvalidReceipt(
                "receipt schema/capability classification was altered".into(),
            ));
        }
        validate_sha256(&self.campaign_manifest_sha256, "campaign manifest")?;
        validate_sha256(&self.native_admission_sha256, "native admission")?;
        validate_declaration_structure(&self.acquisition_declarations)?;
        self.native_admission.validate()?;
        if self.native_admission.capability_classification != NATIVE_CAPABILITY_CLASSIFICATION {
            return Err(SourceBoundAdmissionError::InvalidReceipt(
                "inner native admission capability classification was altered".into(),
            ));
        }
        if self.native_admission.campaign_manifest_sha256 != self.campaign_manifest_sha256 {
            return Err(SourceBoundAdmissionError::InvalidReceipt(
                "inner native admission belongs to a different campaign".into(),
            ));
        }
        let expected_v0 = v0_declarations(&self.acquisition_declarations);
        if self.native_admission.acquisition_declarations != expected_v0 {
            return Err(SourceBoundAdmissionError::InnerDeclarationMismatch);
        }
        let expected = self.native_admission.sha256()?;
        if self.native_admission_sha256 != expected {
            return Err(SourceBoundAdmissionError::InnerAdmissionDigestMismatch {
                expected,
                actual: self.native_admission_sha256.clone(),
            });
        }
        Ok(())
    }

    pub fn validate_with_inputs(
        &self,
        manifest: &Tier1CampaignManifest,
        dossier: &NativeEnvelopeDossier,
    ) -> Result<(), SourceBoundAdmissionError> {
        self.validate()?;
        let recomputed =
            admit_source_bound_native(manifest, dossier, self.acquisition_declarations.clone())?;
        if recomputed != *self {
            return Err(SourceBoundAdmissionError::ReceiptReplayMismatch);
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, SourceBoundAdmissionError> {
        self.validate()?;
        domain_hash(NATIVE_DIGEST_DOMAIN, self)
    }
}

pub fn admit_source_bound_compatibility(
    manifest: &Tier1CampaignManifest,
    dossier: &CandidateVersionBoundDossier,
    declarations: Vec<SourceBoundAcquisitionDeclaration>,
) -> Result<SourceBoundCompatibilityAdmissionReceipt, SourceBoundAdmissionError> {
    let declarations = canonical_source_bound_declarations(manifest, declarations)?;
    let compatibility_admission =
        admit_campaign_result(manifest, dossier, v0_declarations(&declarations))?;
    let receipt = SourceBoundCompatibilityAdmissionReceipt {
        schema: "symthaea.energy-material.source-bound-compatibility-admission.v1".into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        campaign_manifest_sha256: manifest.sha256()?,
        acquisition_declarations: declarations,
        compatibility_admission_sha256: compatibility_admission.sha256()?,
        compatibility_admission,
    };
    receipt.validate()?;
    Ok(receipt)
}

pub fn admit_source_bound_native(
    manifest: &Tier1CampaignManifest,
    dossier: &NativeEnvelopeDossier,
    declarations: Vec<SourceBoundAcquisitionDeclaration>,
) -> Result<SourceBoundNativeAdmissionReceipt, SourceBoundAdmissionError> {
    let declarations = canonical_source_bound_declarations(manifest, declarations)?;
    let native_admission =
        admit_native_campaign_result(manifest, dossier, v0_declarations(&declarations))?;
    let receipt = SourceBoundNativeAdmissionReceipt {
        schema: "symthaea.energy-material.source-bound-native-admission.v1".into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        campaign_manifest_sha256: manifest.sha256()?,
        acquisition_declarations: declarations,
        native_admission_sha256: native_admission.sha256()?,
        native_admission,
    };
    receipt.validate()?;
    Ok(receipt)
}

fn canonical_source_bound_declarations(
    manifest: &Tier1CampaignManifest,
    mut declarations: Vec<SourceBoundAcquisitionDeclaration>,
) -> Result<Vec<SourceBoundAcquisitionDeclaration>, SourceBoundAdmissionError> {
    manifest.validate()?;
    declarations.sort_by_key(|declaration| dimension_code(declaration.dimension));
    validate_declaration_structure(&declarations)?;

    let mut seen = BTreeSet::new();
    for declaration in &declarations {
        let code = dimension_code(declaration.dimension);
        if !seen.insert(code) {
            return Err(SourceBoundAdmissionError::DuplicateDeclaration(
                declaration.dimension,
            ));
        }
        let lane = manifest
            .evidence_lanes
            .iter()
            .find(|lane| lane.dimension == declaration.dimension)
            .ok_or(SourceBoundAdmissionError::MissingCampaignLane(
                declaration.dimension,
            ))?;
        match &lane.source_commitment {
            SourceCommitment::ProspectiveAcquisition {
                source_name,
                source_version,
                source_uri,
                acquisition_query_sha256,
            } => {
                if declaration.source.source_name != *source_name
                    || declaration.source.source_version != *source_version
                    || declaration.source.source_uri != *source_uri
                {
                    return Err(SourceBoundAdmissionError::AcquisitionSourceIdentityMismatch(
                        declaration.dimension,
                    ));
                }
                if declaration.acquisition_query_sha256 != *acquisition_query_sha256 {
                    return Err(SourceBoundAdmissionError::AcquisitionQueryMismatch(
                        declaration.dimension,
                    ));
                }
            }
            _ => {
                return Err(SourceBoundAdmissionError::DeclarationForNonProspectiveLane(
                    declaration.dimension,
                ));
            }
        }
    }

    for lane in &manifest.evidence_lanes {
        if matches!(
            &lane.source_commitment,
            SourceCommitment::ProspectiveAcquisition { .. }
        ) && !seen.contains(&dimension_code(lane.dimension))
        {
            return Err(SourceBoundAdmissionError::MissingDeclaration(lane.dimension));
        }
    }
    Ok(declarations)
}

fn validate_declaration_structure(
    declarations: &[SourceBoundAcquisitionDeclaration],
) -> Result<(), SourceBoundAdmissionError> {
    let mut previous = None;
    let mut seen = BTreeSet::new();
    for declaration in declarations {
        declaration.validate()?;
        let code = dimension_code(declaration.dimension);
        if previous.is_some_and(|prior| code <= prior) {
            return Err(SourceBoundAdmissionError::InvalidReceipt(
                "source-bound declarations must be in canonical dimension order".into(),
            ));
        }
        previous = Some(code);
        if !seen.insert(code) {
            return Err(SourceBoundAdmissionError::DuplicateDeclaration(
                declaration.dimension,
            ));
        }
    }
    Ok(())
}

fn validate_compatibility_inner(
    admission: &CampaignAdmissionReceipt,
) -> Result<(), SourceBoundAdmissionError> {
    if admission.schema != "symthaea.energy-material.campaign-admission.v0"
        || admission.capability_classification != COMPATIBILITY_CAPABILITY_CLASSIFICATION
    {
        return Err(SourceBoundAdmissionError::InvalidReceipt(
            "inner compatibility admission schema/capability classification was altered".into(),
        ));
    }
    for (name, digest) in [
        ("compatibility campaign manifest", admission.campaign_manifest_sha256.as_str()),
        ("candidate-bound dossier", admission.candidate_bound_dossier_sha256.as_str()),
        ("candidate", admission.candidate_sha256.as_str()),
        ("screening policy", admission.screening_policy_sha256.as_str()),
    ] {
        validate_sha256(digest, name)?;
    }
    if admission.admitted_dimensions.as_slice() != EvidenceDimension::ALL.as_slice() {
        return Err(SourceBoundAdmissionError::InvalidReceipt(
            "inner compatibility admission must contain all seven canonical dimensions".into(),
        ));
    }
    Ok(())
}

fn v0_declarations(
    declarations: &[SourceBoundAcquisitionDeclaration],
) -> Vec<AcquisitionDeclaration> {
    declarations
        .iter()
        .map(SourceBoundAcquisitionDeclaration::to_v0)
        .collect()
}

#[derive(Debug, Error)]
pub enum SourceBoundAdmissionError {
    #[error("invalid source-bound acquisition declaration: {0}")]
    InvalidDeclaration(String),
    #[error("invalid source-bound admission receipt: {0}")]
    InvalidReceipt(String),
    #[error("campaign has no lane for dimension {0:?}")]
    MissingCampaignLane(EvidenceDimension),
    #[error("missing source-bound declaration for prospective dimension {0:?}")]
    MissingDeclaration(EvidenceDimension),
    #[error("duplicate source-bound declaration for dimension {0:?}")]
    DuplicateDeclaration(EvidenceDimension),
    #[error("source-bound declaration supplied for non-prospective dimension {0:?}")]
    DeclarationForNonProspectiveLane(EvidenceDimension),
    #[error("acquired source name/version/URI differs from frozen campaign for dimension {0:?}")]
    AcquisitionSourceIdentityMismatch(EvidenceDimension),
    #[error("acquisition query differs from frozen campaign for dimension {0:?}")]
    AcquisitionQueryMismatch(EvidenceDimension),
    #[error("inner admission acquisition declarations differ from source-bound downgrade")]
    InnerDeclarationMismatch,
    #[error("inner admission digest mismatch: expected {expected}, got {actual}")]
    InnerAdmissionDigestMismatch { expected: String, actual: String },
    #[error("source-bound admission receipt does not replay exactly from supplied inputs")]
    ReceiptReplayMismatch,
    #[error(transparent)]
    Campaign(#[from] CampaignError),
    #[error(transparent)]
    NativeAdmission(#[from] NativeAdmissionError),
    #[error("source-bound admission JSON failed to parse/serialize: {0}")]
    Json(#[from] serde_json::Error),
}

fn validate_sha256(value: &str, name: &str) -> Result<(), SourceBoundAdmissionError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(SourceBoundAdmissionError::InvalidDeclaration(format!(
            "{name} SHA-256 must be exactly 64 hexadecimal characters"
        )));
    }
    Ok(())
}

fn dimension_code(dimension: EvidenceDimension) -> u8 {
    match dimension {
        EvidenceDimension::FunctionalPerformance => 0,
        EvidenceDimension::ThermodynamicStability => 1,
        EvidenceDimension::CriticalMaterialBurden => 2,
        EvidenceDimension::SupplyResilience => 3,
        EvidenceDimension::HumanEnvironmentalHazard => 4,
        EvidenceDimension::Circularity => 5,
        EvidenceDimension::Manufacturability => 6,
    }
}

fn domain_hash<T: Serialize>(domain: &[u8], value: &T) -> Result<String, SourceBoundAdmissionError> {
    let bytes = serde_json::to_vec(value)?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    Ok(hex_lower(&hasher.finalize()))
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}
