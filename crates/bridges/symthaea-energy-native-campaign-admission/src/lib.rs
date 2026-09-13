// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Native Tier-1 campaign admission.
//!
//! This is the no-compatibility-attestation path for newly generated evidence.
//! It accepts only dossiers assembled from native candidate/campaign/lane-bound
//! envelopes, revalidates the frozen campaign checks, and records complete
//! negative/infeasible results instead of filtering them out.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_discovery::{EvidenceKind, FidelityLevel, Prediction};
use symthaea_energy_evidence_envelope::EnergyEvidenceEnvelope;
use symthaea_energy_material_campaign::{
    AcquisitionDeclaration, EvidenceLanePlan, SourceCommitment, Tier1CampaignManifest,
};
use symthaea_energy_material_screening::{EvidenceCompleteness, EvidenceDimension};
use symthaea_energy_native_dossier::NativeEnvelopeDossier;
use thiserror::Error;

pub const CAPABILITY_CLASSIFICATION: &str =
    "NATIVE TIER-1 CAMPAIGN ADMISSION ONLY -- complete provenance-plan conformance is not scientific validation, candidate promotion, certification, synthesis authority, investment approval, or deployment authority.";

const ADMISSION_DIGEST_DOMAIN: &[u8] = b"symthaea.energy-material.native-campaign-admission.v0\0";

/// One admitted envelope with its evidence dimension preserved explicitly.
///
/// The dimension and digest are one atomic receipt fact. Keeping them together
/// prevents a valid set of seven digests from losing the provenance mapping
/// that says which exact receipt supplied which Tier-1 dimension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmittedEnvelopeBinding {
    pub dimension: EvidenceDimension,
    pub envelope_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeCampaignAdmissionReceipt {
    pub schema: String,
    pub capability_classification: String,
    pub campaign_manifest_sha256: String,
    pub native_dossier_sha256: String,
    pub candidate_sha256: String,
    pub screening_policy_sha256: String,
    pub admitted_envelopes: Vec<AdmittedEnvelopeBinding>,
    #[serde(default)]
    pub acquisition_declarations: Vec<AcquisitionDeclaration>,
    /// Human-readable disclosure that admission is independent of feasibility.
    pub outcome_disclosure: String,
}

impl NativeCampaignAdmissionReceipt {
    pub fn validate(&self) -> Result<(), NativeAdmissionError> {
        if self.schema != "symthaea.energy-material.native-campaign-admission.v0"
            || self.capability_classification != CAPABILITY_CLASSIFICATION
        {
            return Err(NativeAdmissionError::InvalidAdmission(
                "admission schema/capability classification was altered".into(),
            ));
        }
        for (name, digest) in [
            ("campaign manifest", self.campaign_manifest_sha256.as_str()),
            ("native dossier", self.native_dossier_sha256.as_str()),
            ("candidate", self.candidate_sha256.as_str()),
            ("screening policy", self.screening_policy_sha256.as_str()),
        ] {
            validate_sha256(digest, name)?;
        }
        if self.admitted_envelopes.len() != EvidenceDimension::ALL.len() {
            return Err(NativeAdmissionError::InvalidAdmission(
                "native Tier-1 admission requires seven dimension-bound envelopes".into(),
            ));
        }
        let mut dimensions = BTreeSet::new();
        let mut envelope_digests = BTreeSet::new();
        let mut previous_code = None;
        for binding in &self.admitted_envelopes {
            let code = dimension_code(binding.dimension);
            if previous_code.is_some_and(|previous| code <= previous) {
                return Err(NativeAdmissionError::InvalidAdmission(
                    "admitted envelope bindings must be in canonical dimension order".into(),
                ));
            }
            previous_code = Some(code);
            if !dimensions.insert(code) {
                return Err(NativeAdmissionError::InvalidAdmission(
                    "admitted envelope dimensions contain duplicates".into(),
                ));
            }
            validate_sha256(&binding.envelope_sha256, "native envelope")?;
            if !envelope_digests.insert(binding.envelope_sha256.as_str()) {
                return Err(NativeAdmissionError::InvalidAdmission(
                    "native envelope digest list contains duplicates".into(),
                ));
            }
        }
        if self.outcome_disclosure.trim().is_empty() {
            return Err(NativeAdmissionError::InvalidAdmission(
                "outcome disclosure cannot be empty".into(),
            ));
        }
        validate_acquisition_declarations(&self.acquisition_declarations)?;
        Ok(())
    }

    /// Recompute the complete admission decision from its frozen source inputs.
    ///
    /// This is stronger than structural validation: every lane/source check is
    /// rerun and the resulting receipt must be byte-semantically identical to
    /// this serialized receipt.
    pub fn validate_with_inputs(
        &self,
        manifest: &Tier1CampaignManifest,
        native_dossier: &NativeEnvelopeDossier,
    ) -> Result<(), NativeAdmissionError> {
        self.validate()?;
        let recomputed = admit_native_campaign_result(
            manifest,
            native_dossier,
            self.acquisition_declarations.clone(),
        )?;
        if recomputed != *self {
            return Err(NativeAdmissionError::ReceiptReplayMismatch);
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, NativeAdmissionError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)?;
        let mut hasher = Sha256::new();
        hasher.update(ADMISSION_DIGEST_DOMAIN);
        hasher.update(bytes);
        Ok(hex_lower(&hasher.finalize()))
    }
}

/// Admit a complete native-envelope campaign.
///
/// This deliberately does not require `Feasibility::Feasible`. A complete
/// result that falsifies or disqualifies the candidate remains part of the
/// scientific record.
pub fn admit_native_campaign_result(
    manifest: &Tier1CampaignManifest,
    native_dossier: &NativeEnvelopeDossier,
    mut acquisition_declarations: Vec<AcquisitionDeclaration>,
) -> Result<NativeCampaignAdmissionReceipt, NativeAdmissionError> {
    manifest.validate()?;
    native_dossier.validate_with_manifest(manifest)?;

    if native_dossier.dossier.screening_assessment.completeness != EvidenceCompleteness::Complete {
        return Err(NativeAdmissionError::IncompleteDossier);
    }
    if native_dossier.envelopes.len() != EvidenceDimension::ALL.len()
        || native_dossier.dossier.contributions.len() != EvidenceDimension::ALL.len()
    {
        return Err(NativeAdmissionError::IncompleteDossier);
    }

    // Caller ordering is not scientific evidence. Canonicalize first so the
    // public admission API accepts equivalent declaration sets while serialized
    // receipts still have exactly one deterministic ordering.
    acquisition_declarations.sort_by_key(|declaration| dimension_code(declaration.dimension));
    validate_acquisition_declarations(&acquisition_declarations)?;
    let mut declarations = BTreeMap::new();
    for declaration in &acquisition_declarations {
        let code = dimension_code(declaration.dimension);
        if declarations.insert(code, declaration).is_some() {
            return Err(NativeAdmissionError::DuplicateAcquisitionDeclaration(
                declaration.dimension,
            ));
        }
    }

    let envelopes: BTreeMap<u8, &EnergyEvidenceEnvelope> = native_dossier
        .envelopes
        .iter()
        .map(|envelope| (dimension_code(envelope.binding.dimension), envelope))
        .collect();
    if envelopes.len() != EvidenceDimension::ALL.len() {
        return Err(NativeAdmissionError::IncompleteDossier);
    }

    let mut admitted_envelopes = Vec::with_capacity(EvidenceDimension::ALL.len());
    let mut prospective_dimensions = BTreeSet::new();

    for lane in &manifest.evidence_lanes {
        let envelope = envelopes
            .get(&dimension_code(lane.dimension))
            .copied()
            .ok_or(NativeAdmissionError::MissingDimension(lane.dimension))?;
        envelope.validate_with_manifest(manifest)?;
        validate_prediction_against_lane(lane, &envelope.prediction, manifest)?;

        match &lane.source_commitment {
            SourceCommitment::PinnedEvidenceDigest { sha256 }
            | SourceCommitment::InternalLineage { sha256 } => {
                if !prediction_mentions_sha256(&envelope.prediction, sha256) {
                    return Err(NativeAdmissionError::SourceCommitmentMismatch(
                        lane.dimension,
                    ));
                }
            }
            SourceCommitment::ProspectiveAcquisition {
                acquisition_query_sha256,
                ..
            } => {
                let code = dimension_code(lane.dimension);
                prospective_dimensions.insert(code);
                let declaration = declarations
                    .get(&code)
                    .copied()
                    .ok_or(NativeAdmissionError::MissingAcquisitionDeclaration(
                        lane.dimension,
                    ))?;
                if declaration.acquisition_query_sha256 != *acquisition_query_sha256 {
                    return Err(NativeAdmissionError::AcquisitionQueryMismatch(
                        lane.dimension,
                    ));
                }
                let envelope_sha = envelope.sha256()?;
                if declaration.source_receipt_sha256 != envelope_sha {
                    return Err(NativeAdmissionError::AcquisitionReceiptMismatch(
                        lane.dimension,
                    ));
                }
                if !prediction_mentions_sha256(
                    &envelope.prediction,
                    &declaration.acquired_artifact_sha256,
                ) {
                    return Err(NativeAdmissionError::AcquiredArtifactNotInProvenance(
                        lane.dimension,
                    ));
                }
            }
        }
        admitted_envelopes.push(AdmittedEnvelopeBinding {
            dimension: lane.dimension,
            envelope_sha256: envelope.sha256()?,
        });
    }

    for code in declarations.keys() {
        if !prospective_dimensions.contains(code) {
            return Err(NativeAdmissionError::UnexpectedAcquisitionDeclaration);
        }
    }

    admitted_envelopes.sort_by_key(|binding| dimension_code(binding.dimension));

    let receipt = NativeCampaignAdmissionReceipt {
        schema: "symthaea.energy-material.native-campaign-admission.v0".into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        campaign_manifest_sha256: manifest.sha256()?,
        native_dossier_sha256: native_dossier.sha256(manifest)?,
        candidate_sha256: manifest.candidate_anchor.candidate_sha256.clone(),
        screening_policy_sha256: manifest.screening_policy_sha256.clone(),
        admitted_envelopes,
        acquisition_declarations,
        outcome_disclosure: "Admission records complete conformance to the frozen Tier-1 evidence campaign checks implemented by this crate. Candidate feasibility may be feasible, infeasible, or unknown; admission is not candidate promotion.".into(),
    };
    receipt.validate()?;
    Ok(receipt)
}

fn validate_prediction_against_lane(
    lane: &EvidenceLanePlan,
    prediction: &Prediction,
    manifest: &Tier1CampaignManifest,
) -> Result<(), NativeAdmissionError> {
    prediction.validate()?;
    let contract = manifest
        .screening_policy
        .contracts
        .iter()
        .find(|contract| contract.dimension == lane.dimension)
        .ok_or(NativeAdmissionError::MissingDimension(lane.dimension))?;
    if prediction.metric != contract.metric || prediction.unit != contract.unit {
        return Err(NativeAdmissionError::MetricMismatch(lane.dimension));
    }
    if prediction.fidelity.rank() < contract.minimum_fidelity.rank() {
        return Err(NativeAdmissionError::FidelityBelowPolicy {
            dimension: lane.dimension,
            required: contract.minimum_fidelity,
            found: prediction.fidelity,
        });
    }
    if prediction.model.name != lane.expected_model_name
        || prediction.model.version != lane.expected_model_version
    {
        return Err(NativeAdmissionError::ModelIdentityMismatch(lane.dimension));
    }
    for required in &lane.required_evidence_kinds {
        if !prediction
            .evidence
            .iter()
            .any(|evidence| evidence.kind == *required)
        {
            return Err(NativeAdmissionError::MissingRequiredEvidenceKind {
                dimension: lane.dimension,
                kind: *required,
            });
        }
    }
    Ok(())
}

fn validate_acquisition_declarations(
    declarations: &[AcquisitionDeclaration],
) -> Result<(), NativeAdmissionError> {
    let mut dimensions = BTreeSet::new();
    let mut previous_code = None;
    for declaration in declarations {
        let code = dimension_code(declaration.dimension);
        if previous_code.is_some_and(|previous| code <= previous) {
            return Err(NativeAdmissionError::InvalidAdmission(
                "acquisition declarations must be in canonical dimension order".into(),
            ));
        }
        previous_code = Some(code);
        if !dimensions.insert(code) {
            return Err(NativeAdmissionError::DuplicateAcquisitionDeclaration(
                declaration.dimension,
            ));
        }
        validate_sha256(
            &declaration.acquisition_query_sha256,
            "acquisition query",
        )?;
        validate_sha256(
            &declaration.acquired_artifact_sha256,
            "acquired artifact",
        )?;
        validate_sha256(
            &declaration.source_receipt_sha256,
            "acquisition source receipt",
        )?;
        if declaration.reviewer.trim().is_empty() || declaration.note.trim().is_empty() {
            return Err(NativeAdmissionError::InvalidAdmission(
                "acquisition reviewer and note must be non-empty".into(),
            ));
        }
    }
    Ok(())
}

fn prediction_mentions_sha256(prediction: &Prediction, sha256: &str) -> bool {
    let exact = format!("sha256:{sha256}");
    let model_digests = [
        prediction.model.implementation_digest.as_deref(),
        prediction.model.input_digest.as_deref(),
        prediction.model.output_digest.as_deref(),
    ];
    if model_digests
        .into_iter()
        .flatten()
        .any(|digest| digest.eq_ignore_ascii_case(&exact))
    {
        return true;
    }
    prediction.evidence.iter().any(|evidence| {
        evidence
            .digest
            .as_deref()
            .is_some_and(|digest| digest.eq_ignore_ascii_case(&exact))
    })
}

#[derive(Debug, Error)]
pub enum NativeAdmissionError {
    #[error("invalid native campaign admission: {0}")]
    InvalidAdmission(String),
    #[error("native admission receipt does not replay exactly from the supplied manifest/dossier")]
    ReceiptReplayMismatch,
    #[error("native dossier is not complete across all Tier-1 dimensions")]
    IncompleteDossier,
    #[error("native campaign is missing dimension {0:?}")]
    MissingDimension(EvidenceDimension),
    #[error("metric/unit differs from frozen campaign for dimension {0:?}")]
    MetricMismatch(EvidenceDimension),
    #[error("model identity differs from frozen campaign for dimension {0:?}")]
    ModelIdentityMismatch(EvidenceDimension),
    #[error("fidelity below policy for {dimension:?}: required {required:?}, found {found:?}")]
    FidelityBelowPolicy {
        dimension: EvidenceDimension,
        required: FidelityLevel,
        found: FidelityLevel,
    },
    #[error("missing required evidence kind {kind:?} for dimension {dimension:?}")]
    MissingRequiredEvidenceKind {
        dimension: EvidenceDimension,
        kind: EvidenceKind,
    },
    #[error("source commitment is absent from prediction provenance for dimension {0:?}")]
    SourceCommitmentMismatch(EvidenceDimension),
    #[error("missing prospective acquisition declaration for dimension {0:?}")]
    MissingAcquisitionDeclaration(EvidenceDimension),
    #[error("duplicate acquisition declaration for dimension {0:?}")]
    DuplicateAcquisitionDeclaration(EvidenceDimension),
    #[error("prospective acquisition query differs from frozen campaign for dimension {0:?}")]
    AcquisitionQueryMismatch(EvidenceDimension),
    #[error("acquisition declaration source receipt is not the exact native envelope for dimension {0:?}")]
    AcquisitionReceiptMismatch(EvidenceDimension),
    #[error("acquired artifact digest is absent from prediction provenance for dimension {0:?}")]
    AcquiredArtifactNotInProvenance(EvidenceDimension),
    #[error("acquisition declaration supplied for a non-prospective lane")]
    UnexpectedAcquisitionDeclaration,
    #[error(transparent)]
    Discovery(#[from] symthaea_discovery::DiscoveryError),
    #[error(transparent)]
    Campaign(#[from] symthaea_energy_material_campaign::CampaignError),
    #[error(transparent)]
    Envelope(#[from] symthaea_energy_evidence_envelope::EnvelopeError),
    #[error(transparent)]
    NativeDossier(#[from] symthaea_energy_native_dossier::NativeDossierError),
    #[error("native-admission JSON failed to parse/serialize: {0}")]
    Json(#[from] serde_json::Error),
}

fn validate_sha256(value: &str, name: &str) -> Result<(), NativeAdmissionError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(NativeAdmissionError::InvalidAdmission(format!(
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

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(test)]
mod tests {
    // Full end-to-end campaign fixtures live more naturally in the native
    // dossier/envelope crates. Keep critical invariants here as pure unit tests.
    use super::*;

    #[test]
    fn prediction_digest_search_accepts_model_or_evidence_provenance() {
        use symthaea_discovery::{EvidenceRef, ModelProvenance, UncertaintyEstimate};
        let digest = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let mut prediction = Prediction {
            metric: "fixture".into(),
            value: 1.0,
            unit: "score".into(),
            uncertainty: UncertaintyEstimate::new(1.0, 0.0).unwrap(),
            fidelity: FidelityLevel::Surrogate,
            model: ModelProvenance::named("fixture").unwrap(),
            assumptions: vec![],
            evidence: vec![],
        };
        prediction.model.input_digest = Some(format!("sha256:{digest}"));
        assert!(prediction_mentions_sha256(&prediction, digest));
        prediction.model.input_digest = None;
        prediction.evidence.push(EvidenceRef {
            id: "fixture-evidence".into(),
            kind: EvidenceKind::Dataset,
            uri: None,
            digest: Some(format!("sha256:{digest}")),
            note: None,
        });
        assert!(prediction_mentions_sha256(&prediction, digest));
    }

    #[test]
    fn admission_receipt_requires_all_seven_dimension_bound_envelopes() {
        let receipt = NativeCampaignAdmissionReceipt {
            schema: "symthaea.energy-material.native-campaign-admission.v0".into(),
            capability_classification: CAPABILITY_CLASSIFICATION.into(),
            campaign_manifest_sha256:
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
            native_dossier_sha256:
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into(),
            candidate_sha256:
                "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".into(),
            screening_policy_sha256:
                "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".into(),
            admitted_envelopes: vec![AdmittedEnvelopeBinding {
                dimension: EvidenceDimension::FunctionalPerformance,
                envelope_sha256:
                    "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".into(),
            }],
            acquisition_declarations: vec![],
            outcome_disclosure: "fixture".into(),
        };
        assert!(receipt.validate().is_err());
    }

    #[test]
    fn admission_receipt_rejects_noncanonical_envelope_order() {
        let mut bindings = EvidenceDimension::ALL
            .into_iter()
            .enumerate()
            .map(|(index, dimension)| AdmittedEnvelopeBinding {
                dimension,
                envelope_sha256: format!("{:064x}", index + 1),
            })
            .collect::<Vec<_>>();
        bindings.swap(0, 1);
        let receipt = NativeCampaignAdmissionReceipt {
            schema: "symthaea.energy-material.native-campaign-admission.v0".into(),
            capability_classification: CAPABILITY_CLASSIFICATION.into(),
            campaign_manifest_sha256:
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
            native_dossier_sha256:
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into(),
            candidate_sha256:
                "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".into(),
            screening_policy_sha256:
                "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".into(),
            admitted_envelopes: bindings,
            acquisition_declarations: vec![],
            outcome_disclosure: "fixture".into(),
        };
        assert!(receipt.validate().is_err());
    }
}