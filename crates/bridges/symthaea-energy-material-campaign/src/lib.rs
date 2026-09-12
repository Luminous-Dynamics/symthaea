// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Frozen Tier-1 energy-material campaign manifests and result admission.
//!
//! A complete evidence stack can still be biased after the fact by changing the
//! supply-chain stage, hazard policy, recovery process, source dataset, or model
//! configuration after early results are visible. This crate freezes those
//! choices before result assembly and admits a candidate-bound dossier only when
//! all seven evidence lanes match the declared campaign.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_discovery::{EvidenceKind, FidelityLevel, Prediction};
use symthaea_energy_material_candidate_version::{
    CandidateVersionAnchor, CandidateVersionBoundDossier,
};
use symthaea_energy_material_screening::{
    EnergyMaterialScreeningPolicy, EvidenceCompleteness, EvidenceDimension,
};
use thiserror::Error;

pub const CAPABILITY_CLASSIFICATION: &str =
    "TIER-1 ENERGY-MATERIAL CAMPAIGN MANIFEST ONLY -- preregistered methods and evidence admission are not scientific validation, certification, synthesis authority, or deployment approval.";

const MANIFEST_DIGEST_DOMAIN: &[u8] = b"symthaea.energy-material.campaign-manifest.v0\0";
const ADMISSION_DIGEST_DOMAIN: &[u8] = b"symthaea.energy-material.campaign-admission.v0\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceCommitment {
    /// Exact source/evidence artifact known when the campaign is frozen.
    PinnedEvidenceDigest { sha256: String },
    /// External source will be acquired later using a frozen acquisition/query
    /// specification. Admission then requires a matching acquisition declaration
    /// and evidence provenance mentioning the acquired artifact digest.
    ProspectiveAcquisition {
        source_name: String,
        source_version: String,
        source_uri: String,
        acquisition_query_sha256: String,
    },
    /// Internal model/data lineage known at preregistration time.
    InternalLineage { sha256: String },
}

impl SourceCommitment {
    fn validate(&self) -> Result<(), CampaignError> {
        match self {
            Self::PinnedEvidenceDigest { sha256 } | Self::InternalLineage { sha256 } => {
                validate_sha256(sha256, "source commitment SHA-256")?;
            }
            Self::ProspectiveAcquisition {
                source_name,
                source_version,
                source_uri,
                acquisition_query_sha256,
            } => {
                if source_name.trim().is_empty()
                    || source_version.trim().is_empty()
                    || source_uri.trim().is_empty()
                {
                    return Err(CampaignError::InvalidManifest(
                        "prospective source name/version/URI must be non-empty".into(),
                    ));
                }
                validate_sha256(
                    acquisition_query_sha256,
                    "acquisition-query SHA-256",
                )?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceLanePlan {
    pub dimension: EvidenceDimension,
    pub adapter_name: String,
    pub adapter_version: String,
    pub expected_model_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_model_version: Option<String>,
    #[serde(default)]
    pub method_parameters: BTreeMap<String, String>,
    pub source_commitment: SourceCommitment,
    #[serde(default)]
    pub required_evidence_kinds: Vec<EvidenceKind>,
}

impl EvidenceLanePlan {
    fn validate(&self) -> Result<(), CampaignError> {
        for (name, value) in [
            ("adapter_name", self.adapter_name.as_str()),
            ("adapter_version", self.adapter_version.as_str()),
            ("expected_model_name", self.expected_model_name.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(CampaignError::InvalidManifest(format!(
                    "{name} cannot be empty"
                )));
            }
        }
        if self
            .expected_model_version
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(CampaignError::InvalidManifest(
                "expected_model_version cannot be blank when present".into(),
            ));
        }
        for (key, value) in &self.method_parameters {
            if key.trim().is_empty() || value.trim().is_empty() {
                return Err(CampaignError::InvalidManifest(
                    "method-parameter keys and values must be non-empty".into(),
                ));
            }
        }
        if self.required_evidence_kinds.is_empty() {
            return Err(CampaignError::InvalidManifest(format!(
                "lane {:?} requires at least one evidence kind",
                self.dimension
            )));
        }
        let mut kinds = BTreeSet::new();
        for kind in &self.required_evidence_kinds {
            if !kinds.insert(evidence_kind_code(*kind)) {
                return Err(CampaignError::InvalidManifest(format!(
                    "lane {:?} repeats evidence kind {kind:?}",
                    self.dimension
                )));
            }
        }
        self.source_commitment.validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tier1CampaignManifest {
    pub schema: String,
    pub capability_classification: String,
    pub campaign_id: String,
    pub candidate_anchor: CandidateVersionAnchor,
    pub screening_policy: EnergyMaterialScreeningPolicy,
    pub screening_policy_sha256: String,
    pub evidence_lanes: Vec<EvidenceLanePlan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_registration_reference: Option<String>,
    #[serde(default)]
    pub notes: Vec<String>,
}

impl Tier1CampaignManifest {
    pub fn validate(&self) -> Result<(), CampaignError> {
        if self.schema != "symthaea.energy-material.campaign-manifest.v0"
            || self.capability_classification != CAPABILITY_CLASSIFICATION
        {
            return Err(CampaignError::InvalidManifest(
                "campaign schema/capability classification was altered".into(),
            ));
        }
        if self.campaign_id.trim().is_empty() {
            return Err(CampaignError::InvalidManifest(
                "campaign_id cannot be empty".into(),
            ));
        }
        if self
            .external_registration_reference
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(CampaignError::InvalidManifest(
                "external registration reference cannot be blank when present".into(),
            ));
        }
        if self.notes.iter().any(|note| note.trim().is_empty()) {
            return Err(CampaignError::InvalidManifest(
                "campaign notes cannot contain blank entries".into(),
            ));
        }
        self.candidate_anchor.validate()?;
        self.screening_policy.validate()?;
        let expected_policy_sha = self.screening_policy.sha256()?;
        validate_sha256(&self.screening_policy_sha256, "screening-policy SHA-256")?;
        if self.screening_policy_sha256 != expected_policy_sha {
            return Err(CampaignError::PolicyDigestMismatch {
                expected: expected_policy_sha,
                actual: self.screening_policy_sha256.clone(),
            });
        }
        if self.evidence_lanes.len() != EvidenceDimension::ALL.len() {
            return Err(CampaignError::InvalidManifest(format!(
                "Tier-1 campaign requires exactly {} evidence lanes, got {}",
                EvidenceDimension::ALL.len(),
                self.evidence_lanes.len()
            )));
        }

        let contracts: BTreeMap<u8, _> = self
            .screening_policy
            .contracts
            .iter()
            .map(|contract| (dimension_code(contract.dimension), contract))
            .collect();
        let mut seen = BTreeSet::new();
        for lane in &self.evidence_lanes {
            lane.validate()?;
            let code = dimension_code(lane.dimension);
            if !seen.insert(code) {
                return Err(CampaignError::DuplicateLane(lane.dimension));
            }
            let contract = contracts
                .get(&code)
                .copied()
                .ok_or(CampaignError::MissingPolicyContract(lane.dimension))?;
            for kind in &lane.required_evidence_kinds {
                if !contract.accepted_evidence_kinds.contains(kind) {
                    return Err(CampaignError::EvidenceKindOutsidePolicy {
                        dimension: lane.dimension,
                        kind: *kind,
                    });
                }
            }
        }
        for required in EvidenceDimension::ALL {
            if !seen.contains(&dimension_code(required)) {
                return Err(CampaignError::MissingLane(required));
            }
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, CampaignError> {
        self.validate()?;
        let mut canonical = self.clone();
        canonical
            .evidence_lanes
            .sort_by_key(|lane| dimension_code(lane.dimension));
        for lane in &mut canonical.evidence_lanes {
            lane.required_evidence_kinds
                .sort_by_key(|kind| evidence_kind_code(*kind));
        }
        let bytes = serde_json::to_vec(&canonical)?;
        let mut hasher = Sha256::new();
        hasher.update(MANIFEST_DIGEST_DOMAIN);
        hasher.update(bytes);
        Ok(hex_lower(&hasher.finalize()))
    }
}

pub fn freeze_campaign_manifest(
    campaign_id: impl Into<String>,
    candidate_anchor: CandidateVersionAnchor,
    screening_policy: EnergyMaterialScreeningPolicy,
    mut evidence_lanes: Vec<EvidenceLanePlan>,
    external_registration_reference: Option<String>,
    notes: Vec<String>,
) -> Result<Tier1CampaignManifest, CampaignError> {
    candidate_anchor.validate()?;
    screening_policy.validate()?;
    evidence_lanes.sort_by_key(|lane| dimension_code(lane.dimension));
    let screening_policy_sha256 = screening_policy.sha256()?;
    let manifest = Tier1CampaignManifest {
        schema: "symthaea.energy-material.campaign-manifest.v0".into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        campaign_id: campaign_id.into(),
        candidate_anchor,
        screening_policy,
        screening_policy_sha256,
        evidence_lanes,
        external_registration_reference,
        notes,
    };
    manifest.validate()?;
    Ok(manifest)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcquisitionDeclaration {
    pub dimension: EvidenceDimension,
    pub acquisition_query_sha256: String,
    pub acquired_artifact_sha256: String,
    pub source_receipt_sha256: String,
    pub reviewer: String,
    pub note: String,
}

impl AcquisitionDeclaration {
    fn validate(&self) -> Result<(), CampaignError> {
        validate_sha256(
            &self.acquisition_query_sha256,
            "acquisition declaration query SHA-256",
        )?;
        validate_sha256(
            &self.acquired_artifact_sha256,
            "acquired artifact SHA-256",
        )?;
        validate_sha256(
            &self.source_receipt_sha256,
            "acquisition source-receipt SHA-256",
        )?;
        if self.reviewer.trim().is_empty() || self.note.trim().is_empty() {
            return Err(CampaignError::InvalidAdmission(
                "acquisition reviewer and note must be non-empty".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CampaignAdmissionReceipt {
    pub schema: String,
    pub capability_classification: String,
    pub campaign_manifest_sha256: String,
    pub candidate_bound_dossier_sha256: String,
    pub candidate_sha256: String,
    pub screening_policy_sha256: String,
    pub acquisition_declarations: Vec<AcquisitionDeclaration>,
    pub admitted_dimensions: Vec<EvidenceDimension>,
}

impl CampaignAdmissionReceipt {
    pub fn sha256(&self) -> Result<String, CampaignError> {
        validate_sha256(
            &self.campaign_manifest_sha256,
            "campaign-manifest SHA-256",
        )?;
        validate_sha256(
            &self.candidate_bound_dossier_sha256,
            "candidate-bound dossier SHA-256",
        )?;
        let bytes = serde_json::to_vec(self)?;
        let mut hasher = Sha256::new();
        hasher.update(ADMISSION_DIGEST_DOMAIN);
        hasher.update(bytes);
        Ok(hex_lower(&hasher.finalize()))
    }
}

pub fn admit_campaign_result(
    manifest: &Tier1CampaignManifest,
    bound_dossier: &CandidateVersionBoundDossier,
    mut acquisition_declarations: Vec<AcquisitionDeclaration>,
) -> Result<CampaignAdmissionReceipt, CampaignError> {
    manifest.validate()?;
    bound_dossier.validate_integrity()?;
    if bound_dossier.candidate_anchor.candidate_sha256
        != manifest.candidate_anchor.candidate_sha256
    {
        return Err(CampaignError::CandidateVersionMismatch);
    }
    if bound_dossier.dossier.policy_sha256 != manifest.screening_policy_sha256 {
        return Err(CampaignError::PolicyMismatch);
    }
    if bound_dossier.dossier.screening_assessment.completeness != EvidenceCompleteness::Complete {
        return Err(CampaignError::IncompleteDossier);
    }

    let contributions: BTreeMap<u8, _> = bound_dossier
        .dossier
        .contributions
        .iter()
        .map(|contribution| (dimension_code(contribution.dimension), contribution))
        .collect();
    if contributions.len() != EvidenceDimension::ALL.len() {
        return Err(CampaignError::IncompleteDossier);
    }

    let mut declaration_map = BTreeMap::new();
    for declaration in &acquisition_declarations {
        declaration.validate()?;
        let code = dimension_code(declaration.dimension);
        if declaration_map.insert(code, declaration).is_some() {
            return Err(CampaignError::DuplicateAcquisitionDeclaration(
                declaration.dimension,
            ));
        }
    }

    let mut admitted_dimensions = Vec::with_capacity(EvidenceDimension::ALL.len());
    let mut prospective_expected = BTreeSet::new();
    for lane in &manifest.evidence_lanes {
        let contribution = contributions
            .get(&dimension_code(lane.dimension))
            .copied()
            .ok_or(CampaignError::MissingResultDimension(lane.dimension))?;
        validate_lane_prediction(lane, &contribution.prediction, &manifest.screening_policy)?;

        match &lane.source_commitment {
            SourceCommitment::PinnedEvidenceDigest { sha256 }
            | SourceCommitment::InternalLineage { sha256 } => {
                if !prediction_mentions_sha256(&contribution.prediction, sha256) {
                    return Err(CampaignError::SourceCommitmentMismatch(lane.dimension));
                }
            }
            SourceCommitment::ProspectiveAcquisition {
                acquisition_query_sha256,
                ..
            } => {
                prospective_expected.insert(dimension_code(lane.dimension));
                let declaration = declaration_map
                    .get(&dimension_code(lane.dimension))
                    .copied()
                    .ok_or(CampaignError::MissingAcquisitionDeclaration(lane.dimension))?;
                if declaration.acquisition_query_sha256 != *acquisition_query_sha256
                    || declaration.source_receipt_sha256 != contribution.source_receipt_sha256
                {
                    return Err(CampaignError::AcquisitionDeclarationMismatch(
                        lane.dimension,
                    ));
                }
                if !prediction_mentions_sha256(
                    &contribution.prediction,
                    &declaration.acquired_artifact_sha256,
                ) {
                    return Err(CampaignError::AcquiredArtifactNotInProvenance(
                        lane.dimension,
                    ));
                }
            }
        }
        admitted_dimensions.push(lane.dimension);
    }

    for code in declaration_map.keys() {
        if !prospective_expected.contains(code) {
            return Err(CampaignError::UnexpectedAcquisitionDeclaration);
        }
    }

    acquisition_declarations.sort_by_key(|declaration| dimension_code(declaration.dimension));
    admitted_dimensions.sort_by_key(|dimension| dimension_code(*dimension));

    Ok(CampaignAdmissionReceipt {
        schema: "symthaea.energy-material.campaign-admission.v0".into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        campaign_manifest_sha256: manifest.sha256()?,
        candidate_bound_dossier_sha256: bound_dossier.sha256()?,
        candidate_sha256: manifest.candidate_anchor.candidate_sha256.clone(),
        screening_policy_sha256: manifest.screening_policy_sha256.clone(),
        acquisition_declarations,
        admitted_dimensions,
    })
}

fn validate_lane_prediction(
    lane: &EvidenceLanePlan,
    prediction: &Prediction,
    policy: &EnergyMaterialScreeningPolicy,
) -> Result<(), CampaignError> {
    let contract = policy
        .contracts
        .iter()
        .find(|contract| contract.dimension == lane.dimension)
        .ok_or(CampaignError::MissingPolicyContract(lane.dimension))?;
    if prediction.metric != contract.metric || prediction.unit != contract.unit {
        return Err(CampaignError::MetricMismatch(lane.dimension));
    }
    if prediction.fidelity.rank() < contract.minimum_fidelity.rank() {
        return Err(CampaignError::FidelityBelowPolicy {
            dimension: lane.dimension,
            required: contract.minimum_fidelity,
            found: prediction.fidelity,
        });
    }
    if prediction.model.name != lane.expected_model_name
        || prediction.model.version != lane.expected_model_version
    {
        return Err(CampaignError::ModelIdentityMismatch(lane.dimension));
    }
    for required in &lane.required_evidence_kinds {
        if !prediction
            .evidence
            .iter()
            .any(|evidence| evidence.kind == *required)
        {
            return Err(CampaignError::MissingRequiredEvidenceKind {
                dimension: lane.dimension,
                kind: *required,
            });
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
pub enum CampaignError {
    #[error("invalid Tier-1 campaign manifest: {0}")]
    InvalidManifest(String),
    #[error("invalid campaign admission: {0}")]
    InvalidAdmission(String),
    #[error("screening-policy SHA mismatch: expected {expected}, got {actual}")]
    PolicyDigestMismatch { expected: String, actual: String },
    #[error("duplicate evidence lane {0:?}")]
    DuplicateLane(EvidenceDimension),
    #[error("missing evidence lane {0:?}")]
    MissingLane(EvidenceDimension),
    #[error("screening policy has no contract for dimension {0:?}")]
    MissingPolicyContract(EvidenceDimension),
    #[error("lane {dimension:?} requires evidence kind {kind:?} that policy does not accept")]
    EvidenceKindOutsidePolicy {
        dimension: EvidenceDimension,
        kind: EvidenceKind,
    },
    #[error("candidate version differs from frozen campaign")]
    CandidateVersionMismatch,
    #[error("screening policy differs from frozen campaign")]
    PolicyMismatch,
    #[error("candidate-bound dossier is not complete across all Tier-1 dimensions")]
    IncompleteDossier,
    #[error("campaign result is missing dimension {0:?}")]
    MissingResultDimension(EvidenceDimension),
    #[error("metric/unit differs from frozen policy for dimension {0:?}")]
    MetricMismatch(EvidenceDimension),
    #[error("model identity differs from frozen campaign for dimension {0:?}")]
    ModelIdentityMismatch(EvidenceDimension),
    #[error("fidelity below frozen policy for {dimension:?}: required {required:?}, found {found:?}")]
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
    #[error("source commitment is not present in prediction provenance for dimension {0:?}")]
    SourceCommitmentMismatch(EvidenceDimension),
    #[error("missing acquisition declaration for prospective dimension {0:?}")]
    MissingAcquisitionDeclaration(EvidenceDimension),
    #[error("duplicate acquisition declaration for dimension {0:?}")]
    DuplicateAcquisitionDeclaration(EvidenceDimension),
    #[error("acquisition declaration differs from frozen plan/receipt for dimension {0:?}")]
    AcquisitionDeclarationMismatch(EvidenceDimension),
    #[error("acquired artifact digest is absent from prediction provenance for dimension {0:?}")]
    AcquiredArtifactNotInProvenance(EvidenceDimension),
    #[error("acquisition declaration supplied for a non-prospective lane")]
    UnexpectedAcquisitionDeclaration,
    #[error(transparent)]
    CandidateVersion(#[from] symthaea_energy_material_candidate_version::CandidateVersionError),
    #[error(transparent)]
    Screening(#[from] symthaea_energy_material_screening::ScreeningError),
    #[error("campaign JSON failed to parse/serialize: {0}")]
    Json(#[from] serde_json::Error),
}

fn validate_sha256(value: &str, name: &str) -> Result<(), CampaignError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(CampaignError::InvalidManifest(format!(
            "{name} must be exactly 64 hexadecimal characters"
        )));
    }
    Ok(())
}

fn evidence_kind_code(kind: EvidenceKind) -> u8 {
    match kind {
        EvidenceKind::Literature => 0,
        EvidenceKind::Dataset => 1,
        EvidenceKind::Heuristic => 2,
        EvidenceKind::SurrogateModel => 3,
        EvidenceKind::AnalyticalModel => 4,
        EvidenceKind::FirstPrinciplesSimulation => 5,
        EvidenceKind::ExternalSimulation => 6,
        EvidenceKind::Experiment => 7,
        EvidenceKind::IndependentReplication => 8,
        EvidenceKind::DeviceValidation => 9,
        EvidenceKind::FieldObservation => 10,
    }
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
    use super::*;
    use std::collections::BTreeMap;
    use symthaea_discovery::{
        Candidate, CandidateId, CandidateOrigin, ObjectiveDirection,
    };
    use symthaea_energy_material_candidate_version::anchor_candidate;
    use symthaea_energy_material_screening::MetricContract;

    fn anchor() -> CandidateVersionAnchor {
        let mut specification = BTreeMap::new();
        specification.insert("formula".into(), "LiFePO4".into());
        anchor_candidate(Candidate {
            id: CandidateId::new("candidate-a").unwrap(),
            kind: "energy_material".into(),
            specification,
            origin: CandidateOrigin::UserProposed,
        })
        .unwrap()
    }

    fn policy() -> EnergyMaterialScreeningPolicy {
        EnergyMaterialScreeningPolicy {
            policy_id: "campaign-fixture".into(),
            contracts: EvidenceDimension::ALL
                .into_iter()
                .map(|dimension| MetricContract {
                    dimension,
                    metric: format!("metric-{}", dimension_code(dimension)),
                    unit: "score".into(),
                    direction: ObjectiveDirection::Minimize,
                    minimum_fidelity: FidelityLevel::Surrogate,
                    accepted_evidence_kinds: vec![EvidenceKind::Dataset],
                })
                .collect(),
            constraints: vec![],
        }
    }

    fn lanes() -> Vec<EvidenceLanePlan> {
        EvidenceDimension::ALL
            .into_iter()
            .map(|dimension| EvidenceLanePlan {
                dimension,
                adapter_name: format!("adapter-{}", dimension_code(dimension)),
                adapter_version: "v0".into(),
                expected_model_name: format!("model-{}", dimension_code(dimension)),
                expected_model_version: Some("v0".into()),
                method_parameters: BTreeMap::new(),
                source_commitment: SourceCommitment::InternalLineage {
                    sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                        .into(),
                },
                required_evidence_kinds: vec![EvidenceKind::Dataset],
            })
            .collect()
    }

    #[test]
    fn campaign_requires_all_seven_lanes() {
        let mut incomplete = lanes();
        incomplete.pop();
        assert!(freeze_campaign_manifest(
            "campaign-a",
            anchor(),
            policy(),
            incomplete,
            None,
            vec![]
        )
        .is_err());
    }

    #[test]
    fn lane_input_order_does_not_change_manifest_digest() {
        let first = freeze_campaign_manifest(
            "campaign-a",
            anchor(),
            policy(),
            lanes(),
            None,
            vec![],
        )
        .unwrap();
        let mut reversed = lanes();
        reversed.reverse();
        let second = freeze_campaign_manifest(
            "campaign-a",
            anchor(),
            policy(),
            reversed,
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(first.sha256().unwrap(), second.sha256().unwrap());
    }

    #[test]
    fn source_method_changes_manifest_identity() {
        let first = freeze_campaign_manifest(
            "campaign-a",
            anchor(),
            policy(),
            lanes(),
            None,
            vec![],
        )
        .unwrap();
        let mut changed = lanes();
        changed[0]
            .method_parameters
            .insert("aggregation".into(), "maximum".into());
        let second = freeze_campaign_manifest(
            "campaign-a",
            anchor(),
            policy(),
            changed,
            None,
            vec![],
        )
        .unwrap();
        assert_ne!(first.sha256().unwrap(), second.sha256().unwrap());
    }

    #[test]
    fn policy_rejects_lane_evidence_kind_it_does_not_accept() {
        let mut changed = lanes();
        changed[0].required_evidence_kinds = vec![EvidenceKind::Experiment];
        assert!(matches!(
            freeze_campaign_manifest(
                "campaign-a",
                anchor(),
                policy(),
                changed,
                None,
                vec![]
            ),
            Err(CampaignError::EvidenceKindOutsidePolicy { .. })
        ));
    }
}
