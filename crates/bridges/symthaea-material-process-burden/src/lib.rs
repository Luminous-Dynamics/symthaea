// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit material-process burden evidence.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use symthaea_discovery::{
    CandidateId, EvidenceKind, EvidenceRef, FidelityLevel, ModelProvenance, Prediction,
    UncertaintyEstimate,
};
use symthaea_energy_material_screening::{metric, unit};
use thiserror::Error;

pub const CAPABILITY_CLASSIFICATION: &str =
    "MATERIAL PROCESS-BURDEN EVIDENCE ONLY -- process temperature and step count are not a universal manufacturability score, cost model, qualification, or deployment authority.";
const DATASET_DIGEST_DOMAIN: &[u8] = b"symthaea.material-process-burden.dataset.v0\0";
const RECEIPT_DIGEST_DOMAIN: &[u8] = b"symthaea.material-process-burden.receipt.v0\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessEvidenceBasis {
    Modeled,
    DocumentedExperimental,
    Pilot,
    Industrial,
}

impl ProcessEvidenceBasis {
    fn fidelity(self) -> FidelityLevel {
        match self {
            Self::Modeled => FidelityLevel::Analytical,
            Self::DocumentedExperimental | Self::Pilot | Self::Industrial => FidelityLevel::Experiment,
        }
    }

    fn evidence_kind(self) -> EvidenceKind {
        match self {
            Self::Modeled => EvidenceKind::AnalyticalModel,
            Self::DocumentedExperimental | Self::Pilot | Self::Industrial => EvidenceKind::Experiment,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessRecord {
    pub material_id: String,
    pub process_id: String,
    pub process_name: String,
    pub maximum_process_temperature_k: f64,
    pub synthesis_step_count: u32,
    pub basis: ProcessEvidenceBasis,
    pub conditions_note: String,
    pub last_updated: String,
}

impl ProcessRecord {
    fn validate(&self) -> Result<(), ProcessError> {
        for (name, value) in [
            ("material_id", self.material_id.as_str()),
            ("process_id", self.process_id.as_str()),
            ("process_name", self.process_name.as_str()),
            ("conditions_note", self.conditions_note.as_str()),
            ("last_updated", self.last_updated.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(ProcessError::InvalidDataset(format!("{name} cannot be blank")));
            }
        }
        if !self.maximum_process_temperature_k.is_finite() || self.maximum_process_temperature_k <= 0.0 {
            return Err(ProcessError::InvalidDataset(
                "maximum process temperature must be finite and positive".into(),
            ));
        }
        if self.synthesis_step_count == 0 {
            return Err(ProcessError::InvalidDataset(
                "synthesis step count must be positive".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessDataset {
    pub source_title: String,
    pub source_version: String,
    pub source_uri: String,
    pub source_document_sha256: String,
    pub normalized_capture_sha256: String,
    pub normalized_at_utc: String,
    pub extraction_note: String,
    pub records: Vec<ProcessRecord>,
}

#[derive(Debug, Deserialize)]
struct RawProcessDataset {
    source_title: String,
    source_version: String,
    source_uri: String,
    source_document_sha256: String,
    normalized_at_utc: String,
    extraction_note: String,
    records: Vec<ProcessRecord>,
}

impl ProcessDataset {
    pub fn from_json_bytes(raw_json: &[u8]) -> Result<Self, ProcessError> {
        let raw: RawProcessDataset = serde_json::from_slice(raw_json)?;
        let mut records = raw.records;
        records.sort_by(|a, b| a.material_id.cmp(&b.material_id).then_with(|| a.process_id.cmp(&b.process_id)));
        let dataset = Self {
            source_title: raw.source_title,
            source_version: raw.source_version,
            source_uri: raw.source_uri,
            source_document_sha256: raw.source_document_sha256,
            normalized_capture_sha256: sha256_bytes(raw_json),
            normalized_at_utc: raw.normalized_at_utc,
            extraction_note: raw.extraction_note,
            records,
        };
        dataset.validate()?;
        Ok(dataset)
    }

    pub fn validate(&self) -> Result<(), ProcessError> {
        for (name, value) in [
            ("source_title", self.source_title.as_str()),
            ("source_version", self.source_version.as_str()),
            ("source_uri", self.source_uri.as_str()),
            ("normalized_at_utc", self.normalized_at_utc.as_str()),
            ("extraction_note", self.extraction_note.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(ProcessError::InvalidDataset(format!("{name} cannot be blank")));
            }
        }
        for digest in [&self.source_document_sha256, &self.normalized_capture_sha256] {
            if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(ProcessError::InvalidDataset(
                    "source/capture SHA-256 must be 64 hexadecimal characters".into(),
                ));
            }
        }
        if self.records.is_empty() {
            return Err(ProcessError::InvalidDataset("process dataset cannot be empty".into()));
        }
        let mut keys = BTreeSet::new();
        for record in &self.records {
            record.validate()?;
            if !keys.insert((record.material_id.as_str(), record.process_id.as_str())) {
                return Err(ProcessError::InvalidDataset(format!(
                    "duplicate material/process pair {}/{}",
                    record.material_id, record.process_id
                )));
            }
        }
        Ok(())
    }

    pub fn verify_source_document(&self, source_document: &[u8]) -> Result<(), ProcessError> {
        self.validate()?;
        let actual = sha256_bytes(source_document);
        if !actual.eq_ignore_ascii_case(&self.source_document_sha256) {
            return Err(ProcessError::SourceDocumentDigestMismatch {
                expected: self.source_document_sha256.clone(),
                actual,
            });
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, ProcessError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)?;
        let mut hasher = Sha256::new();
        hasher.update(DATASET_DIGEST_DOMAIN);
        hasher.update(bytes);
        Ok(hex_lower(&hasher.finalize()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessBindingBasis {
    ExactMaterialId,
    ExplicitMapping { note: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessBinding {
    pub candidate_id: CandidateId,
    pub material_id: String,
    pub process_id: String,
    pub basis: ProcessBindingBasis,
}

impl ProcessBinding {
    fn validate(&self) -> Result<(), ProcessError> {
        CandidateId::new(self.candidate_id.0.clone())?;
        if self.material_id.trim().is_empty() || self.process_id.trim().is_empty() {
            return Err(ProcessError::InvalidBinding(
                "material_id and process_id cannot be blank".into(),
            ));
        }
        match &self.basis {
            ProcessBindingBasis::ExactMaterialId if self.candidate_id.0 != self.material_id => {
                return Err(ProcessError::InvalidBinding(
                    "ExactMaterialId requires candidate_id == material_id".into(),
                ));
            }
            ProcessBindingBasis::ExplicitMapping { note } if note.trim().is_empty() => {
                return Err(ProcessError::InvalidBinding(
                    "explicit mapping requires a non-empty note".into(),
                ));
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessBurdenReceipt {
    pub schema: String,
    pub capability_classification: String,
    pub binding: ProcessBinding,
    pub process_name: String,
    pub conditions_note: String,
    pub evidence_basis: ProcessEvidenceBasis,
    pub source_document_sha256: String,
    pub normalized_capture_sha256: String,
    pub process_dataset_sha256: String,
    pub predictions: Vec<Prediction>,
}

impl ProcessBurdenReceipt {
    pub fn sha256(&self) -> Result<String, ProcessError> {
        let bytes = serde_json::to_vec(self)?;
        let mut hasher = Sha256::new();
        hasher.update(RECEIPT_DIGEST_DOMAIN);
        hasher.update(bytes);
        Ok(hex_lower(&hasher.finalize()))
    }
}

pub fn bind_process_burden_from_bytes(
    normalized_dataset_json: &[u8],
    source_document: &[u8],
    binding: ProcessBinding,
) -> Result<ProcessBurdenReceipt, ProcessError> {
    let dataset = ProcessDataset::from_json_bytes(normalized_dataset_json)?;
    dataset.verify_source_document(source_document)?;
    binding.validate()?;
    let record = dataset
        .records
        .iter()
        .find(|record| record.material_id == binding.material_id && record.process_id == binding.process_id)
        .ok_or_else(|| ProcessError::RecordNotFound {
            material_id: binding.material_id.clone(),
            process_id: binding.process_id.clone(),
        })?;

    let dataset_sha256 = dataset.sha256()?;
    let binding_assumption = match &binding.basis {
        ProcessBindingBasis::ExactMaterialId => {
            "Candidate identity exactly matches the source material identifier.".to_owned()
        }
        ProcessBindingBasis::ExplicitMapping { note } => format!(
            "Candidate-to-material identity is caller-declared, not independently proven: {note}"
        ),
    };

    let mut model = ModelProvenance::named("material process-burden extraction")?;
    model.version = Some(format!("v0;process={};basis={:?}", record.process_id, record.basis));
    model.input_digest = Some(format!("sha256:{dataset_sha256}"));

    let evidence = vec![
        EvidenceRef {
            id: format!("process-dataset:{}:{}", record.material_id, record.process_id),
            kind: EvidenceKind::Dataset,
            uri: Some(dataset.source_uri.clone()),
            digest: Some(format!("sha256:{}", dataset.source_document_sha256)),
            note: Some(format!("normalized capture sha256={}", dataset.normalized_capture_sha256)),
        },
        EvidenceRef {
            id: format!("process-basis:{}:{}", record.material_id, record.process_id),
            kind: record.basis.evidence_kind(),
            uri: Some(dataset.source_uri.clone()),
            digest: Some(format!("sha256:{}", dataset.source_document_sha256)),
            note: Some(format!(
                "basis={:?}; conditions={}; last_updated={}",
                record.basis, record.conditions_note, record.last_updated
            )),
        },
    ];
    let assumptions = vec![
        "Maximum process temperature and step count are separate measurable process-burden quantities; they are not combined into a universal manufacturability score.".into(),
        "These values apply only to the exact declared material/process/conditions and do not establish yield, purity, throughput, equipment availability, cost, scale-up success, worker safety, or product quality.".into(),
        "No calibrated predictive uncertainty is supplied by this adapter; epistemic uncertainty is marked fully unknown.".into(),
        binding_assumption,
    ];

    let temperature = Prediction {
        metric: metric::MAXIMUM_PROCESS_TEMPERATURE.into(),
        value: record.maximum_process_temperature_k,
        unit: unit::KELVIN.into(),
        uncertainty: UncertaintyEstimate::new(1.0, 0.0)?,
        fidelity: record.basis.fidelity(),
        model: model.clone(),
        assumptions: assumptions.clone(),
        evidence: evidence.clone(),
    };
    let steps = Prediction {
        metric: metric::SYNTHESIS_STEP_COUNT.into(),
        value: f64::from(record.synthesis_step_count),
        unit: unit::COUNT.into(),
        uncertainty: UncertaintyEstimate::new(1.0, 0.0)?,
        fidelity: record.basis.fidelity(),
        model,
        assumptions,
        evidence,
    };
    temperature.validate()?;
    steps.validate()?;

    Ok(ProcessBurdenReceipt {
        schema: "symthaea.material-process-burden.receipt.v0".into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        binding,
        process_name: record.process_name.clone(),
        conditions_note: record.conditions_note.clone(),
        evidence_basis: record.basis,
        source_document_sha256: dataset.source_document_sha256.clone(),
        normalized_capture_sha256: dataset.normalized_capture_sha256.clone(),
        process_dataset_sha256: dataset_sha256,
        predictions: vec![temperature, steps],
    })
}

#[derive(Debug, Error)]
pub enum ProcessError {
    #[error("invalid process dataset: {0}")]
    InvalidDataset(String),
    #[error("process source document SHA-256 mismatch: expected {expected}, got {actual}")]
    SourceDocumentDigestMismatch { expected: String, actual: String },
    #[error("invalid candidate/material/process binding: {0}")]
    InvalidBinding(String),
    #[error("process record not found for material {material_id:?}, process {process_id:?}")]
    RecordNotFound { material_id: String, process_id: String },
    #[error(transparent)]
    Discovery(#[from] symthaea_discovery::DiscoveryError),
    #[error("process JSON failed to parse/serialize: {0}")]
    Json(#[from] serde_json::Error),
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
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

    const SOURCE: &[u8] = b"fixture process source";

    fn dataset_json(basis: ProcessEvidenceBasis) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "source_title": "Fixture Synthesis Procedure",
            "source_version": "2026-fixture",
            "source_uri": "https://example.invalid/process",
            "source_document_sha256": sha256_bytes(SOURCE),
            "normalized_at_utc": "2026-09-12T00:00:00Z",
            "extraction_note": "test fixture only",
            "records": [{
                "material_id": "material-1",
                "process_id": "route-a",
                "process_name": "fixture solid-state route",
                "maximum_process_temperature_k": 1123.0,
                "synthesis_step_count": 4,
                "basis": basis,
                "conditions_note": "fixture atmosphere and dwell conditions",
                "last_updated": "2026-09-01"
            }]
        }))
        .unwrap()
    }

    fn binding() -> ProcessBinding {
        ProcessBinding {
            candidate_id: CandidateId::new("candidate-a").unwrap(),
            material_id: "material-1".into(),
            process_id: "route-a".into(),
            basis: ProcessBindingBasis::ExplicitMapping {
                note: "fixture material identity assertion".into(),
            },
        }
    }

    #[test]
    fn emits_separate_temperature_and_step_predictions_without_composite_score() {
        let receipt = bind_process_burden_from_bytes(
            &dataset_json(ProcessEvidenceBasis::DocumentedExperimental),
            SOURCE,
            binding(),
        )
        .unwrap();
        assert_eq!(receipt.predictions.len(), 2);
        assert_eq!(receipt.predictions[0].metric, metric::MAXIMUM_PROCESS_TEMPERATURE);
        assert_eq!(receipt.predictions[1].metric, metric::SYNTHESIS_STEP_COUNT);
        assert!(receipt.predictions.iter().all(|p| p.fidelity == FidelityLevel::Experiment));
    }

    #[test]
    fn modeled_process_does_not_masquerade_as_experiment() {
        let receipt = bind_process_burden_from_bytes(
            &dataset_json(ProcessEvidenceBasis::Modeled),
            SOURCE,
            binding(),
        )
        .unwrap();
        assert!(receipt.predictions.iter().all(|p| p.fidelity == FidelityLevel::Analytical));
    }

    #[test]
    fn invalid_process_values_and_wrong_source_fail_closed() {
        let mut value: serde_json::Value = serde_json::from_slice(&dataset_json(
            ProcessEvidenceBasis::DocumentedExperimental,
        ))
        .unwrap();
        value["records"][0]["synthesis_step_count"] = serde_json::json!(0);
        assert!(ProcessDataset::from_json_bytes(&serde_json::to_vec(&value).unwrap()).is_err());

        assert!(matches!(
            bind_process_burden_from_bytes(
                &dataset_json(ProcessEvidenceBasis::DocumentedExperimental),
                b"wrong source",
                binding(),
            ),
            Err(ProcessError::SourceDocumentDigestMismatch { .. })
        ));
    }
}
