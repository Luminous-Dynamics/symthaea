// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Native candidate/campaign lineage envelopes for Tier-1 energy-material evidence.
//!
//! Existing adapter receipts can be wrapped without rewriting their payload
//! schema. The outer receipt cryptographically commits to the exact candidate
//! version, frozen campaign, evidence lane, dimension, generic prediction,
//! payload type, and exact UTF-8 JSON receipt text. This does not prove the
//! adapter obeyed the lane's method parameters, but it makes the new receipt's
//! claimed lineage and normalized discovery result immutable and machine-checkable.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use symthaea_discovery::{CandidateId, EvidenceKind, Prediction};
use symthaea_energy_material_campaign::{EvidenceLanePlan, Tier1CampaignManifest};
use symthaea_energy_material_screening::EvidenceDimension;
use thiserror::Error;

pub const CAPABILITY_CLASSIFICATION: &str =
    "ENERGY EVIDENCE LINEAGE ENVELOPE ONLY -- content-addressed candidate/campaign/prediction binding is not proof of method adherence, scientific validation, certification, or deployment authority.";

const LANE_DIGEST_DOMAIN: &[u8] = b"symthaea.energy-material.campaign-lane.v0\0";
const PREDICTION_DIGEST_DOMAIN: &[u8] = b"symthaea.energy-material.generic-prediction.v0\0";
const PAYLOAD_DIGEST_DOMAIN: &[u8] = b"symthaea.energy-material.evidence-payload.v0\0";
const ENVELOPE_DIGEST_DOMAIN: &[u8] = b"symthaea.energy-material.evidence-envelope.v0\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeCampaignBinding {
    pub candidate_id: CandidateId,
    pub candidate_sha256: String,
    pub campaign_manifest_sha256: String,
    pub campaign_lane_sha256: String,
    pub dimension: EvidenceDimension,
}

impl NativeCampaignBinding {
    pub fn validate(&self) -> Result<(), EnvelopeError> {
        CandidateId::new(self.candidate_id.0.clone())?;
        validate_sha256(&self.candidate_sha256, "candidate SHA-256")?;
        validate_sha256(
            &self.campaign_manifest_sha256,
            "campaign-manifest SHA-256",
        )?;
        validate_sha256(&self.campaign_lane_sha256, "campaign-lane SHA-256")?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyEvidenceEnvelope {
    pub schema: String,
    pub capability_classification: String,
    pub binding: NativeCampaignBinding,
    /// Exact generic discovery prediction promoted into the dossier layer.
    pub prediction: Prediction,
    pub prediction_sha256: String,
    pub payload_type: String,
    /// Exact UTF-8 JSON adapter receipt text. Whitespace and key order are
    /// intentionally evidence-bearing in this native outer envelope.
    pub payload_json: String,
    pub payload_sha256: String,
}

impl EnergyEvidenceEnvelope {
    pub fn validate(&self) -> Result<(), EnvelopeError> {
        if self.schema != "symthaea.energy-material.evidence-envelope.v0"
            || self.capability_classification != CAPABILITY_CLASSIFICATION
        {
            return Err(EnvelopeError::InvalidEnvelope(
                "envelope schema/capability classification was altered".into(),
            ));
        }
        self.binding.validate()?;
        self.prediction.validate()?;
        validate_sha256(&self.prediction_sha256, "prediction SHA-256")?;
        let expected_prediction = prediction_sha256(&self.prediction)?;
        if self.prediction_sha256 != expected_prediction {
            return Err(EnvelopeError::PredictionDigestMismatch {
                expected: expected_prediction,
                actual: self.prediction_sha256.clone(),
            });
        }
        if self.payload_type.trim().is_empty() {
            return Err(EnvelopeError::InvalidEnvelope(
                "payload_type cannot be empty".into(),
            ));
        }
        if self.payload_json.is_empty() {
            return Err(EnvelopeError::InvalidEnvelope(
                "payload JSON cannot be empty".into(),
            ));
        }
        let parsed: serde_json::Value = serde_json::from_str(&self.payload_json)?;
        if parsed.is_null() {
            return Err(EnvelopeError::InvalidEnvelope(
                "payload cannot be JSON null".into(),
            ));
        }
        validate_sha256(&self.payload_sha256, "payload SHA-256")?;
        let expected_payload = payload_sha256(self.payload_json.as_bytes());
        if self.payload_sha256 != expected_payload {
            return Err(EnvelopeError::PayloadDigestMismatch {
                expected: expected_payload,
                actual: self.payload_sha256.clone(),
            });
        }
        Ok(())
    }

    pub fn payload_value(&self) -> Result<serde_json::Value, EnvelopeError> {
        self.validate()?;
        serde_json::from_str(&self.payload_json).map_err(EnvelopeError::Json)
    }

    pub fn validate_with_manifest(
        &self,
        manifest: &Tier1CampaignManifest,
    ) -> Result<(), EnvelopeError> {
        self.validate()?;
        let expected = binding_from_manifest(manifest, self.binding.dimension)?;
        if self.binding != expected {
            return Err(EnvelopeError::CampaignBindingMismatch(
                self.binding.dimension,
            ));
        }
        let contract = manifest
            .screening_policy
            .contracts
            .iter()
            .find(|contract| contract.dimension == self.binding.dimension)
            .ok_or(EnvelopeError::MissingCampaignLane(self.binding.dimension))?;
        if self.prediction.metric != contract.metric || self.prediction.unit != contract.unit {
            return Err(EnvelopeError::PredictionContractMismatch(
                self.binding.dimension,
            ));
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, EnvelopeError> {
        self.validate()?;
        let encoded = serde_json::to_vec(self)?;
        let mut hasher = Sha256::new();
        hasher.update(ENVELOPE_DIGEST_DOMAIN);
        hasher.update(encoded);
        Ok(hex_lower(&hasher.finalize()))
    }
}

pub fn wrap_evidence_payload_json(
    manifest: &Tier1CampaignManifest,
    dimension: EvidenceDimension,
    prediction: Prediction,
    payload_type: impl Into<String>,
    payload_json: impl Into<String>,
) -> Result<EnergyEvidenceEnvelope, EnvelopeError> {
    manifest.validate()?;
    prediction.validate()?;
    let payload_type = payload_type.into();
    if payload_type.trim().is_empty() {
        return Err(EnvelopeError::InvalidEnvelope(
            "payload_type cannot be empty".into(),
        ));
    }
    let payload_json = payload_json.into();
    if payload_json.is_empty() {
        return Err(EnvelopeError::InvalidEnvelope(
            "payload JSON cannot be empty".into(),
        ));
    }
    let parsed: serde_json::Value = serde_json::from_str(&payload_json)?;
    if parsed.is_null() {
        return Err(EnvelopeError::InvalidEnvelope(
            "payload cannot be JSON null".into(),
        ));
    }
    let binding = binding_from_manifest(manifest, dimension)?;
    let prediction_sha256 = prediction_sha256(&prediction)?;
    let payload_sha256 = payload_sha256(payload_json.as_bytes());
    let envelope = EnergyEvidenceEnvelope {
        schema: "symthaea.energy-material.evidence-envelope.v0".into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        binding,
        prediction,
        prediction_sha256,
        payload_type,
        payload_json,
        payload_sha256,
    };
    envelope.validate_with_manifest(manifest)?;
    Ok(envelope)
}

pub fn binding_from_manifest(
    manifest: &Tier1CampaignManifest,
    dimension: EvidenceDimension,
) -> Result<NativeCampaignBinding, EnvelopeError> {
    manifest.validate()?;
    let lane = manifest
        .evidence_lanes
        .iter()
        .find(|lane| lane.dimension == dimension)
        .ok_or(EnvelopeError::MissingCampaignLane(dimension))?;
    Ok(NativeCampaignBinding {
        candidate_id: manifest.candidate_anchor.candidate.id.clone(),
        candidate_sha256: manifest.candidate_anchor.candidate_sha256.clone(),
        campaign_manifest_sha256: manifest.sha256()?,
        campaign_lane_sha256: campaign_lane_sha256(lane)?,
        dimension,
    })
}

pub fn campaign_lane_sha256(lane: &EvidenceLanePlan) -> Result<String, EnvelopeError> {
    // Full campaign construction validates source commitment and screening-policy
    // compatibility. This standalone hash keeps lane identity canonical while
    // remaining usable by adapter receipt builders.
    if lane.adapter_name.trim().is_empty()
        || lane.adapter_version.trim().is_empty()
        || lane.expected_model_name.trim().is_empty()
    {
        return Err(EnvelopeError::InvalidLane(
            "adapter/model names and adapter version must be non-empty".into(),
        ));
    }
    if lane
        .expected_model_version
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(EnvelopeError::InvalidLane(
            "expected model version cannot be blank when present".into(),
        ));
    }
    if lane.required_evidence_kinds.is_empty() {
        return Err(EnvelopeError::InvalidLane(
            "lane requires at least one evidence kind".into(),
        ));
    }
    for (key, value) in &lane.method_parameters {
        if key.trim().is_empty() || value.trim().is_empty() {
            return Err(EnvelopeError::InvalidLane(
                "method parameters require non-empty keys and values".into(),
            ));
        }
    }

    let mut canonical = lane.clone();
    canonical
        .required_evidence_kinds
        .sort_by_key(|kind| evidence_kind_code(*kind));
    let encoded = serde_json::to_vec(&canonical)?;
    let mut hasher = Sha256::new();
    hasher.update(LANE_DIGEST_DOMAIN);
    hasher.update(encoded);
    Ok(hex_lower(&hasher.finalize()))
}

pub fn prediction_sha256(prediction: &Prediction) -> Result<String, EnvelopeError> {
    prediction.validate()?;
    let encoded = serde_json::to_vec(prediction)?;
    let mut hasher = Sha256::new();
    hasher.update(PREDICTION_DIGEST_DOMAIN);
    hasher.update(encoded);
    Ok(hex_lower(&hasher.finalize()))
}

pub fn payload_sha256(payload_json: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(PAYLOAD_DIGEST_DOMAIN);
    hasher.update(payload_json);
    hex_lower(&hasher.finalize())
}

#[derive(Debug, Error)]
pub enum EnvelopeError {
    #[error("invalid evidence envelope: {0}")]
    InvalidEnvelope(String),
    #[error("invalid campaign lane: {0}")]
    InvalidLane(String),
    #[error("campaign has no lane for dimension {0:?}")]
    MissingCampaignLane(EvidenceDimension),
    #[error("prediction SHA-256 mismatch: expected {expected}, got {actual}")]
    PredictionDigestMismatch { expected: String, actual: String },
    #[error("payload SHA-256 mismatch: expected {expected}, got {actual}")]
    PayloadDigestMismatch { expected: String, actual: String },
    #[error("envelope binding differs from frozen campaign for dimension {0:?}")]
    CampaignBindingMismatch(EvidenceDimension),
    #[error("generic prediction metric/unit differs from frozen policy for dimension {0:?}")]
    PredictionContractMismatch(EvidenceDimension),
    #[error(transparent)]
    Discovery(#[from] symthaea_discovery::DiscoveryError),
    #[error(transparent)]
    Campaign(#[from] symthaea_energy_material_campaign::CampaignError),
    #[error("evidence-envelope JSON failed to parse/serialize: {0}")]
    Json(#[from] serde_json::Error),
}

fn validate_sha256(value: &str, name: &str) -> Result<(), EnvelopeError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(EnvelopeError::InvalidEnvelope(format!(
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
        Candidate, CandidateId, CandidateOrigin, EvidenceRef, FidelityLevel, ModelProvenance,
        ObjectiveDirection, UncertaintyEstimate,
    };
    use symthaea_energy_material_campaign::{
        freeze_campaign_manifest, EvidenceLanePlan, SourceCommitment,
    };
    use symthaea_energy_material_candidate_version::anchor_candidate;
    use symthaea_energy_material_screening::{
        EnergyMaterialScreeningPolicy, MetricContract,
    };

    fn manifest() -> Tier1CampaignManifest {
        let candidate = Candidate {
            id: CandidateId::new("candidate-a").unwrap(),
            kind: "energy_material".into(),
            specification: BTreeMap::from([("formula".into(), "LiFePO4".into())]),
            origin: CandidateOrigin::UserProposed,
        };
        let policy = EnergyMaterialScreeningPolicy {
            policy_id: "envelope-test".into(),
            contracts: EvidenceDimension::ALL
                .into_iter()
                .map(|dimension| MetricContract {
                    dimension,
                    metric: format!("metric-{dimension:?}"),
                    unit: "score".into(),
                    direction: ObjectiveDirection::Minimize,
                    minimum_fidelity: FidelityLevel::Surrogate,
                    accepted_evidence_kinds: vec![EvidenceKind::Dataset],
                })
                .collect(),
            constraints: vec![],
        };
        let lanes = EvidenceDimension::ALL
            .into_iter()
            .map(|dimension| EvidenceLanePlan {
                dimension,
                adapter_name: format!("adapter-{dimension:?}"),
                adapter_version: "v0".into(),
                expected_model_name: format!("model-{dimension:?}"),
                expected_model_version: Some("v0".into()),
                method_parameters: BTreeMap::new(),
                source_commitment: SourceCommitment::InternalLineage {
                    sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                        .into(),
                },
                required_evidence_kinds: vec![EvidenceKind::Dataset],
            })
            .collect();
        freeze_campaign_manifest(
            "envelope-campaign",
            anchor_candidate(candidate).unwrap(),
            policy,
            lanes,
            None,
            vec![],
        )
        .unwrap()
    }

    fn prediction(dimension: EvidenceDimension) -> Prediction {
        Prediction {
            metric: format!("metric-{dimension:?}"),
            value: 1.0,
            unit: "score".into(),
            uncertainty: UncertaintyEstimate::new(1.0, 0.0).unwrap(),
            fidelity: FidelityLevel::Surrogate,
            model: ModelProvenance {
                name: format!("model-{dimension:?}"),
                version: Some("v0".into()),
                implementation_digest: None,
                input_digest: Some(
                    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                        .into(),
                ),
                output_digest: None,
            },
            assumptions: vec![],
            evidence: vec![EvidenceRef {
                id: "fixture-dataset".into(),
                kind: EvidenceKind::Dataset,
                uri: None,
                digest: Some(
                    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                        .into(),
                ),
                note: None,
            }],
        }
    }

    #[test]
    fn envelope_binds_candidate_campaign_lane_prediction_and_exact_payload_text() {
        let manifest = manifest();
        let dimension = EvidenceDimension::FunctionalPerformance;
        let prediction = prediction(dimension);
        let payload = "{\"value\":42,\"unit\":\"fixture\"}";
        let envelope = wrap_evidence_payload_json(
            &manifest,
            dimension,
            prediction.clone(),
            "fixture-receipt-v0",
            payload,
        )
        .unwrap();
        envelope.validate_with_manifest(&manifest).unwrap();
        assert_eq!(
            envelope.binding.candidate_sha256,
            manifest.candidate_anchor.candidate_sha256
        );
        assert_eq!(envelope.binding.campaign_manifest_sha256, manifest.sha256().unwrap());
        assert_eq!(envelope.prediction, prediction);
        assert_eq!(envelope.prediction_sha256, prediction_sha256(&envelope.prediction).unwrap());
        assert_eq!(envelope.payload_json, payload);
        assert!(!envelope.binding.campaign_lane_sha256.is_empty());
        assert!(!envelope.sha256().unwrap().is_empty());
    }

    #[test]
    fn changing_lane_method_parameters_changes_lane_digest() {
        let first = manifest();
        let first_lane = first
            .evidence_lanes
            .iter()
            .find(|lane| lane.dimension == EvidenceDimension::FunctionalPerformance)
            .unwrap();
        let first_digest = campaign_lane_sha256(first_lane).unwrap();

        let mut changed = first_lane.clone();
        changed
            .method_parameters
            .insert("aggregation".into(), "different".into());
        assert_ne!(first_digest, campaign_lane_sha256(&changed).unwrap());
    }

    #[test]
    fn exact_json_text_is_evidence_bearing() {
        let manifest = manifest();
        let dimension = EvidenceDimension::FunctionalPerformance;
        let compact = wrap_evidence_payload_json(
            &manifest,
            dimension,
            prediction(dimension),
            "fixture-receipt-v0",
            "{\"a\":1,\"b\":2}",
        )
        .unwrap();
        let reordered = wrap_evidence_payload_json(
            &manifest,
            dimension,
            prediction(dimension),
            "fixture-receipt-v0",
            "{\"b\":2,\"a\":1}",
        )
        .unwrap();
        assert_ne!(compact.payload_sha256, reordered.payload_sha256);
        assert_ne!(compact.sha256().unwrap(), reordered.sha256().unwrap());
    }

    #[test]
    fn prediction_mutation_breaks_validation() {
        let manifest = manifest();
        let dimension = EvidenceDimension::FunctionalPerformance;
        let mut envelope = wrap_evidence_payload_json(
            &manifest,
            dimension,
            prediction(dimension),
            "fixture-receipt-v0",
            "{\"value\":1}",
        )
        .unwrap();
        envelope.prediction.value = 2.0;
        assert!(matches!(
            envelope.validate(),
            Err(EnvelopeError::PredictionDigestMismatch { .. })
        ));
    }

    #[test]
    fn payload_mutation_breaks_validation() {
        let manifest = manifest();
        let dimension = EvidenceDimension::FunctionalPerformance;
        let mut envelope = wrap_evidence_payload_json(
            &manifest,
            dimension,
            prediction(dimension),
            "fixture-receipt-v0",
            "{\"value\":1}",
        )
        .unwrap();
        envelope.payload_json = "{\"value\":2}".into();
        assert!(matches!(
            envelope.validate(),
            Err(EnvelopeError::PayloadDigestMismatch { .. })
        ));
    }

    #[test]
    fn different_campaign_cannot_validate_same_envelope() {
        let manifest = manifest();
        let dimension = EvidenceDimension::FunctionalPerformance;
        let envelope = wrap_evidence_payload_json(
            &manifest,
            dimension,
            prediction(dimension),
            "fixture-receipt-v0",
            "{\"value\":1}",
        )
        .unwrap();
        let mut changed = manifest.clone();
        changed.campaign_id = "other-campaign".into();
        assert!(matches!(
            envelope.validate_with_manifest(&changed),
            Err(EnvelopeError::CampaignBindingMismatch(_))
        ));
    }
}
