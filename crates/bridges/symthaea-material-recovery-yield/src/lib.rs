// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Process-specific material recovery-yield evidence.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use symthaea_discovery::{
    CandidateId, EvidenceKind, EvidenceRef, FidelityLevel, ModelProvenance, Prediction,
    UncertaintyEstimate,
};
use symthaea_energy_material_screening::unit;
use thiserror::Error;

pub const RECOVERY_YIELD_METRIC: &str = "material_recovery_yield_fraction";
pub const CAPABILITY_CLASSIFICATION: &str =
    "PROCESS-SPECIFIC RECOVERY-YIELD EVIDENCE ONLY -- recovery under one declared process/feedstock is not universal recyclability, circularity certification, or deployment authority.";
const DATASET_DIGEST_DOMAIN: &[u8] = b"symthaea.material-recovery.dataset.v0\0";
const RECEIPT_DIGEST_DOMAIN: &[u8] = b"symthaea.material-recovery.receipt.v0\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryEvidenceBasis {
    Modeled,
    MeasuredLab,
    MeasuredPilot,
    MeasuredIndustrial,
}

impl RecoveryEvidenceBasis {
    fn fidelity(self) -> FidelityLevel {
        match self {
            Self::Modeled => FidelityLevel::Analytical,
            Self::MeasuredLab | Self::MeasuredPilot | Self::MeasuredIndustrial => {
                FidelityLevel::Experiment
            }
        }
    }

    fn evidence_kind(self) -> EvidenceKind {
        match self {
            Self::Modeled => EvidenceKind::AnalyticalModel,
            Self::MeasuredLab | Self::MeasuredPilot | Self::MeasuredIndustrial => {
                EvidenceKind::Experiment
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecoveryRecord {
    pub material_id: String,
    pub process_id: String,
    pub feedstock_form: String,
    pub recovered_product: String,
    pub recovery_yield_fraction: f64,
    pub basis: RecoveryEvidenceBasis,
    pub conditions_note: String,
    pub last_updated: String,
}

impl RecoveryRecord {
    fn validate(&self) -> Result<(), RecoveryError> {
        for (name, value) in [
            ("material_id", self.material_id.as_str()),
            ("process_id", self.process_id.as_str()),
            ("feedstock_form", self.feedstock_form.as_str()),
            ("recovered_product", self.recovered_product.as_str()),
            ("conditions_note", self.conditions_note.as_str()),
            ("last_updated", self.last_updated.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(RecoveryError::InvalidDataset(format!("{name} cannot be blank")));
            }
        }
        if !self.recovery_yield_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.recovery_yield_fraction)
        {
            return Err(RecoveryError::InvalidDataset(format!(
                "recovery yield for {}/{} must be in [0,1]",
                self.material_id, self.process_id
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecoveryDataset {
    pub source_title: String,
    pub source_version: String,
    pub source_uri: String,
    pub source_document_sha256: String,
    pub normalized_capture_sha256: String,
    pub normalized_at_utc: String,
    pub extraction_note: String,
    pub records: Vec<RecoveryRecord>,
}

#[derive(Debug, Deserialize)]
struct RawRecoveryDataset {
    source_title: String,
    source_version: String,
    source_uri: String,
    source_document_sha256: String,
    normalized_at_utc: String,
    extraction_note: String,
    records: Vec<RecoveryRecord>,
}

impl RecoveryDataset {
    pub fn from_json_bytes(raw_json: &[u8]) -> Result<Self, RecoveryError> {
        let raw: RawRecoveryDataset = serde_json::from_slice(raw_json)?;
        let mut records = raw.records;
        records.sort_by(|a, b| {
            a.material_id
                .cmp(&b.material_id)
                .then_with(|| a.process_id.cmp(&b.process_id))
        });
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

    pub fn validate(&self) -> Result<(), RecoveryError> {
        for (name, value) in [
            ("source_title", self.source_title.as_str()),
            ("source_version", self.source_version.as_str()),
            ("source_uri", self.source_uri.as_str()),
            ("normalized_at_utc", self.normalized_at_utc.as_str()),
            ("extraction_note", self.extraction_note.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(RecoveryError::InvalidDataset(format!("{name} cannot be blank")));
            }
        }
        for digest in [&self.source_document_sha256, &self.normalized_capture_sha256] {
            if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(RecoveryError::InvalidDataset(
                    "source/capture SHA-256 must be 64 hexadecimal characters".into(),
                ));
            }
        }
        if self.records.is_empty() {
            return Err(RecoveryError::InvalidDataset(
                "recovery dataset must contain at least one record".into(),
            ));
        }
        let mut keys = BTreeSet::new();
        for record in &self.records {
            record.validate()?;
            if !keys.insert((record.material_id.as_str(), record.process_id.as_str())) {
                return Err(RecoveryError::InvalidDataset(format!(
                    "duplicate material/process pair {}/{}",
                    record.material_id, record.process_id
                )));
            }
        }
        Ok(())
    }

    pub fn verify_source_document(&self, source_document: &[u8]) -> Result<(), RecoveryError> {
        self.validate()?;
        let actual = sha256_bytes(source_document);
        if !actual.eq_ignore_ascii_case(&self.source_document_sha256) {
            return Err(RecoveryError::SourceDocumentDigestMismatch {
                expected: self.source_document_sha256.clone(),
                actual,
            });
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, RecoveryError> {
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
pub enum RecoveryBindingBasis {
    ExactMaterialId,
    ExplicitMapping { note: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryBinding {
    pub candidate_id: CandidateId,
    pub material_id: String,
    pub process_id: String,
    pub basis: RecoveryBindingBasis,
}

impl RecoveryBinding {
    fn validate(&self) -> Result<(), RecoveryError> {
        CandidateId::new(self.candidate_id.0.clone())?;
        if self.material_id.trim().is_empty() || self.process_id.trim().is_empty() {
            return Err(RecoveryError::InvalidBinding(
                "material_id and process_id cannot be blank".into(),
            ));
        }
        match &self.basis {
            RecoveryBindingBasis::ExactMaterialId => {
                if self.candidate_id.0 != self.material_id {
                    return Err(RecoveryError::InvalidBinding(
                        "ExactMaterialId requires candidate_id == material_id".into(),
                    ));
                }
            }
            RecoveryBindingBasis::ExplicitMapping { note } if note.trim().is_empty() => {
                return Err(RecoveryError::InvalidBinding(
                    "explicit material mapping requires a non-empty note".into(),
                ));
            }
            RecoveryBindingBasis::ExplicitMapping { .. } => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecoveryYieldReceipt {
    pub schema: String,
    pub capability_classification: String,
    pub binding: RecoveryBinding,
    pub feedstock_form: String,
    pub recovered_product: String,
    pub conditions_note: String,
    pub evidence_basis: RecoveryEvidenceBasis,
    pub source_document_sha256: String,
    pub normalized_capture_sha256: String,
    pub recovery_dataset_sha256: String,
    pub prediction: Prediction,
}

impl RecoveryYieldReceipt {
    pub fn sha256(&self) -> Result<String, RecoveryError> {
        let bytes = serde_json::to_vec(self)?;
        let mut hasher = Sha256::new();
        hasher.update(RECEIPT_DIGEST_DOMAIN);
        hasher.update(bytes);
        Ok(hex_lower(&hasher.finalize()))
    }
}

pub fn bind_recovery_yield_from_bytes(
    normalized_dataset_json: &[u8],
    source_document: &[u8],
    binding: RecoveryBinding,
) -> Result<RecoveryYieldReceipt, RecoveryError> {
    let dataset = RecoveryDataset::from_json_bytes(normalized_dataset_json)?;
    dataset.verify_source_document(source_document)?;
    binding.validate()?;

    let record = dataset
        .records
        .iter()
        .find(|record| {
            record.material_id == binding.material_id && record.process_id == binding.process_id
        })
        .ok_or_else(|| RecoveryError::RecordNotFound {
            material_id: binding.material_id.clone(),
            process_id: binding.process_id.clone(),
        })?;

    let dataset_sha256 = dataset.sha256()?;
    let binding_assumption = match &binding.basis {
        RecoveryBindingBasis::ExactMaterialId => {
            "Candidate identity exactly matches the source material identifier.".to_owned()
        }
        RecoveryBindingBasis::ExplicitMapping { note } => format!(
            "Candidate-to-material identity is caller-declared, not independently proven: {note}"
        ),
    };

    let mut model = ModelProvenance::named("process-specific material recovery yield")?;
    model.version = Some(format!("v0;process={};basis={:?}", record.process_id, record.basis));
    model.input_digest = Some(format!("sha256:{dataset_sha256}"));

    let prediction = Prediction {
        metric: RECOVERY_YIELD_METRIC.into(),
        value: record.recovery_yield_fraction,
        unit: unit::FRACTION.into(),
        uncertainty: UncertaintyEstimate::new(1.0, 0.0)?,
        fidelity: record.basis.fidelity(),
        model,
        assumptions: vec![
            "Recovery yield applies only to the exact declared material, process, feedstock form, recovered product and conditions note.".into(),
            "Process-specific recovery yield is not universal recyclability and does not establish collection rate, product purity, economic viability, energy use, reagent burden, repeated-cycle quality, or closed-loop reuse.".into(),
            "No calibrated predictive uncertainty is supplied by this adapter; epistemic uncertainty is marked fully unknown.".into(),
            binding_assumption,
        ],
        evidence: vec![
            EvidenceRef {
                id: format!("recovery-dataset:{}:{}", record.material_id, record.process_id),
                kind: EvidenceKind::Dataset,
                uri: Some(dataset.source_uri.clone()),
                digest: Some(format!("sha256:{}", dataset.source_document_sha256)),
                note: Some(format!(
                    "normalized capture sha256={}; source version={}",
                    dataset.normalized_capture_sha256, dataset.source_version
                )),
            },
            EvidenceRef {
                id: format!("recovery-basis:{}:{}", record.material_id, record.process_id),
                kind: record.basis.evidence_kind(),
                uri: Some(dataset.source_uri.clone()),
                digest: Some(format!("sha256:{}", dataset.source_document_sha256)),
                note: Some(format!(
                    "basis={:?}; feedstock={}; recovered_product={}; conditions={}; last_updated={}",
                    record.basis,
                    record.feedstock_form,
                    record.recovered_product,
                    record.conditions_note,
                    record.last_updated
                )),
            },
        ],
    };
    prediction.validate()?;

    Ok(RecoveryYieldReceipt {
        schema: "symthaea.material-recovery-yield.receipt.v0".into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        binding,
        feedstock_form: record.feedstock_form.clone(),
        recovered_product: record.recovered_product.clone(),
        conditions_note: record.conditions_note.clone(),
        evidence_basis: record.basis,
        source_document_sha256: dataset.source_document_sha256.clone(),
        normalized_capture_sha256: dataset.normalized_capture_sha256.clone(),
        recovery_dataset_sha256: dataset_sha256,
        prediction,
    })
}

#[derive(Debug, Error)]
pub enum RecoveryError {
    #[error("invalid recovery dataset: {0}")]
    InvalidDataset(String),
    #[error("recovery source document SHA-256 mismatch: expected {expected}, got {actual}")]
    SourceDocumentDigestMismatch { expected: String, actual: String },
    #[error("invalid candidate/material/process binding: {0}")]
    InvalidBinding(String),
    #[error("recovery record not found for material {material_id:?}, process {process_id:?}")]
    RecordNotFound { material_id: String, process_id: String },
    #[error(transparent)]
    Discovery(#[from] symthaea_discovery::DiscoveryError),
    #[error("recovery JSON failed to parse/serialize: {0}")]
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

    const SOURCE: &[u8] = b"fixture recovery source";

    fn dataset_json(basis: RecoveryEvidenceBasis) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "source_title": "Fixture Recovery Study",
            "source_version": "2026-fixture",
            "source_uri": "https://example.invalid/recovery",
            "source_document_sha256": sha256_bytes(SOURCE),
            "normalized_at_utc": "2026-09-12T00:00:00Z",
            "extraction_note": "test fixture only",
            "records": [{
                "material_id": "material-1",
                "process_id": "process-a",
                "feedstock_form": "sorted end-of-life powder",
                "recovered_product": "purified active-material precursor",
                "recovery_yield_fraction": 0.91,
                "basis": basis,
                "conditions_note": "fixture conditions",
                "last_updated": "2026-09-01"
            }]
        }))
        .unwrap()
    }

    fn binding() -> RecoveryBinding {
        RecoveryBinding {
            candidate_id: CandidateId::new("candidate-a").unwrap(),
            material_id: "material-1".into(),
            process_id: "process-a".into(),
            basis: RecoveryBindingBasis::ExplicitMapping {
                note: "fixture material identity assertion".into(),
            },
        }
    }

    #[test]
    fn measured_recovery_is_experiment_fidelity_and_process_bound() {
        let receipt = bind_recovery_yield_from_bytes(
            &dataset_json(RecoveryEvidenceBasis::MeasuredLab),
            SOURCE,
            binding(),
        )
        .unwrap();
        assert_eq!(receipt.prediction.value, 0.91);
        assert_eq!(receipt.prediction.fidelity, FidelityLevel::Experiment);
        assert!(receipt.prediction.evidence.iter().any(|e| e.kind == EvidenceKind::Experiment));
        assert_eq!(receipt.feedstock_form, "sorted end-of-life powder");
    }

    #[test]
    fn modeled_recovery_does_not_masquerade_as_experiment() {
        let receipt = bind_recovery_yield_from_bytes(
            &dataset_json(RecoveryEvidenceBasis::Modeled),
            SOURCE,
            binding(),
        )
        .unwrap();
        assert_eq!(receipt.prediction.fidelity, FidelityLevel::Analytical);
        assert!(receipt.prediction.evidence.iter().any(|e| e.kind == EvidenceKind::AnalyticalModel));
    }

    #[test]
    fn wrong_source_and_wrong_process_fail_closed() {
        assert!(matches!(
            bind_recovery_yield_from_bytes(
                &dataset_json(RecoveryEvidenceBasis::MeasuredLab),
                b"wrong source",
                binding(),
            ),
            Err(RecoveryError::SourceDocumentDigestMismatch { .. })
        ));

        let mut wrong = binding();
        wrong.process_id = "other-process".into();
        assert!(matches!(
            bind_recovery_yield_from_bytes(
                &dataset_json(RecoveryEvidenceBasis::MeasuredLab),
                SOURCE,
                wrong,
            ),
            Err(RecoveryError::RecordNotFound { .. })
        ));
    }

    #[test]
    fn invalid_yield_and_exact_id_mismatch_fail_closed() {
        let mut value: serde_json::Value = serde_json::from_slice(&dataset_json(
            RecoveryEvidenceBasis::MeasuredLab,
        ))
        .unwrap();
        value["records"][0]["recovery_yield_fraction"] = serde_json::json!(1.1);
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(RecoveryDataset::from_json_bytes(&bytes).is_err());

        let exact = RecoveryBinding {
            candidate_id: CandidateId::new("candidate-a").unwrap(),
            material_id: "material-1".into(),
            process_id: "process-a".into(),
            basis: RecoveryBindingBasis::ExactMaterialId,
        };
        assert!(bind_recovery_yield_from_bytes(
            &dataset_json(RecoveryEvidenceBasis::MeasuredLab),
            SOURCE,
            exact,
        )
        .is_err());
    }
}
