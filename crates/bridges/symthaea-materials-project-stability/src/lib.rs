// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Network-free Materials Project thermodynamic-stability evidence adapter.
//!
//! A caller exports Materials Project SummaryDoc JSON through an external client
//! and supplies the exact bytes plus capture metadata. This crate hashes the raw
//! bytes, normalizes only the fields needed for stability evidence, and creates
//! a discovery prediction only after an explicit candidate -> MP material
//! binding is supplied.

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
    "MATERIALS PROJECT STABILITY EVIDENCE ADAPTER ONLY -- computed database evidence is not experimental validation, synthesis success, safety certification, or deployment authority.";
pub const CAPTURE_SCHEMA: &str = "symthaea.materials-project.summary-stability-capture.v0";
pub const SUMMARY_ROUTE: &str = "/materials/summary";

const REQUIRED_FIELDS: [&str; 7] = [
    "material_id",
    "formula_pretty",
    "energy_above_hull",
    "formation_energy_per_atom",
    "deprecated",
    "last_updated",
    "warnings",
];
const CAPTURE_DIGEST_DOMAIN: &[u8] = b"symthaea.materials-project.stability-capture.v0\0";
const RECEIPT_DIGEST_DOMAIN: &[u8] = b"symthaea.materials-project.stability-receipt.v0\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureMetadata {
    pub source_name: String,
    pub api_route: String,
    pub mp_api_version: String,
    pub emmet_version: String,
    pub retrieved_at_utc: String,
    pub query_description: String,
    pub requested_fields: Vec<String>,
}

impl CaptureMetadata {
    pub fn validate(&self) -> Result<(), StabilityError> {
        for (name, value) in [
            ("source_name", self.source_name.as_str()),
            ("api_route", self.api_route.as_str()),
            ("mp_api_version", self.mp_api_version.as_str()),
            ("emmet_version", self.emmet_version.as_str()),
            ("retrieved_at_utc", self.retrieved_at_utc.as_str()),
            ("query_description", self.query_description.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(StabilityError::InvalidMetadata(format!(
                    "{name} cannot be empty"
                )));
            }
        }
        if self.source_name != "Materials Project" || self.api_route != SUMMARY_ROUTE {
            return Err(StabilityError::InvalidMetadata(
                "official capture requires source_name='Materials Project' and api_route='/materials/summary'"
                    .into(),
            ));
        }
        let fields: BTreeSet<&str> = self.requested_fields.iter().map(String::as_str).collect();
        if fields.len() != self.requested_fields.len() {
            return Err(StabilityError::InvalidMetadata(
                "requested_fields contains duplicates".into(),
            ));
        }
        let missing: Vec<&str> = REQUIRED_FIELDS
            .iter()
            .copied()
            .filter(|field| !fields.contains(field))
            .collect();
        if !missing.is_empty() {
            return Err(StabilityError::InvalidMetadata(format!(
                "capture is missing required SummaryDoc fields {missing:?}"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StabilityRecord {
    pub material_id: String,
    pub formula_pretty: String,
    pub energy_above_hull_ev_per_atom: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub formation_energy_per_atom_ev: Option<f64>,
    pub deprecated: bool,
    pub last_updated: String,
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl StabilityRecord {
    fn validate(&self) -> Result<(), StabilityError> {
        if self.material_id.trim().is_empty()
            || self.formula_pretty.trim().is_empty()
            || self.last_updated.trim().is_empty()
        {
            return Err(StabilityError::InvalidRecord(
                "material_id, formula_pretty and last_updated must be non-empty".into(),
            ));
        }
        if !self.energy_above_hull_ev_per_atom.is_finite()
            || self.energy_above_hull_ev_per_atom < 0.0
        {
            return Err(StabilityError::InvalidRecord(format!(
                "{} has invalid energy_above_hull {:?}",
                self.material_id, self.energy_above_hull_ev_per_atom
            )));
        }
        if let Some(value) = self.formation_energy_per_atom_ev {
            if !value.is_finite() {
                return Err(StabilityError::InvalidRecord(format!(
                    "{} has non-finite formation energy",
                    self.material_id
                )));
            }
        }
        if self.warnings.iter().any(|warning| warning.trim().is_empty()) {
            return Err(StabilityError::InvalidRecord(format!(
                "{} contains an empty warning entry",
                self.material_id
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StabilityCapture {
    pub schema: &'static str,
    pub capability_classification: &'static str,
    pub metadata: CaptureMetadata,
    pub raw_json_sha256: String,
    pub records: Vec<StabilityRecord>,
}

impl StabilityCapture {
    pub fn sha256(&self) -> Result<String, StabilityError> {
        self.validate()?;
        let encoded = serde_json::to_vec(self)?;
        let mut hasher = Sha256::new();
        hasher.update(CAPTURE_DIGEST_DOMAIN);
        hasher.update(encoded);
        Ok(hex_lower(&hasher.finalize()))
    }

    pub fn validate(&self) -> Result<(), StabilityError> {
        if self.schema != CAPTURE_SCHEMA || self.capability_classification != CAPABILITY_CLASSIFICATION {
            return Err(StabilityError::InvalidCapture(
                "capture schema/capability classification was altered".into(),
            ));
        }
        self.metadata.validate()?;
        if self.raw_json_sha256.len() != 64
            || !self.raw_json_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(StabilityError::InvalidCapture(
                "raw JSON SHA-256 must be 64 hexadecimal characters".into(),
            ));
        }
        if self.records.is_empty() {
            return Err(StabilityError::InvalidCapture(
                "capture must contain at least one stability record".into(),
            ));
        }
        let mut ids = BTreeSet::new();
        for record in &self.records {
            record.validate()?;
            if !ids.insert(record.material_id.as_str()) {
                return Err(StabilityError::DuplicateMaterialId(
                    record.material_id.clone(),
                ));
            }
        }
        Ok(())
    }
}

/// Raw subset deserialized from SummaryDoc JSON. Unknown SummaryDoc fields are
/// intentionally ignored, while the exact source bytes remain content-addressed.
#[derive(Debug, Deserialize)]
struct RawSummaryRecord {
    material_id: Option<String>,
    formula_pretty: Option<String>,
    energy_above_hull: Option<f64>,
    formation_energy_per_atom: Option<f64>,
    #[serde(default)]
    deprecated: bool,
    last_updated: Option<String>,
    #[serde(default)]
    warnings: Vec<String>,
}

/// Parse a JSON array produced from SummaryDoc `model_dump(mode="json")`
/// objects. This function performs no network access.
pub fn parse_summary_docs_json(
    raw_json: &[u8],
    metadata: CaptureMetadata,
) -> Result<StabilityCapture, StabilityError> {
    metadata.validate()?;
    let raw_records: Vec<RawSummaryRecord> = serde_json::from_slice(raw_json)?;
    if raw_records.is_empty() {
        return Err(StabilityError::InvalidCapture(
            "SummaryDoc JSON array cannot be empty".into(),
        ));
    }

    let raw_json_sha256 = sha256_bytes(raw_json);
    let mut records = Vec::with_capacity(raw_records.len());
    for (index, raw) in raw_records.into_iter().enumerate() {
        let record = StabilityRecord {
            material_id: required_string(raw.material_id, index, "material_id")?,
            formula_pretty: required_string(raw.formula_pretty, index, "formula_pretty")?,
            energy_above_hull_ev_per_atom: raw.energy_above_hull.ok_or_else(|| {
                StabilityError::InvalidRecord(format!(
                    "SummaryDoc row {index} is missing energy_above_hull"
                ))
            })?,
            formation_energy_per_atom_ev: raw.formation_energy_per_atom,
            deprecated: raw.deprecated,
            last_updated: required_string(raw.last_updated, index, "last_updated")?,
            warnings: raw.warnings,
        };
        record.validate()?;
        records.push(record);
    }
    records.sort_by(|left, right| left.material_id.cmp(&right.material_id));

    let capture = StabilityCapture {
        schema: CAPTURE_SCHEMA,
        capability_classification: CAPABILITY_CLASSIFICATION,
        metadata,
        raw_json_sha256,
        records,
    };
    capture.validate()?;
    Ok(capture)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BindingBasis {
    /// Candidate id is exactly the Materials Project material id.
    ExactMaterialId,
    /// Caller asserts why a differently named candidate refers to this exact MP
    /// material/structure. This is a declared mapping, not independently proven
    /// identity.
    ExplicitMapping { note: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StabilityBinding {
    pub candidate_id: CandidateId,
    pub material_id: String,
    pub basis: BindingBasis,
}

impl StabilityBinding {
    pub fn validate(&self) -> Result<(), StabilityError> {
        CandidateId::new(self.candidate_id.0.clone())?;
        if self.material_id.trim().is_empty() {
            return Err(StabilityError::InvalidBinding(
                "material_id cannot be empty".into(),
            ));
        }
        match &self.basis {
            BindingBasis::ExactMaterialId => {
                if self.candidate_id.0 != self.material_id {
                    return Err(StabilityError::InvalidBinding(
                        "ExactMaterialId requires candidate_id == material_id".into(),
                    ));
                }
            }
            BindingBasis::ExplicitMapping { note } => {
                if note.trim().is_empty() {
                    return Err(StabilityError::InvalidBinding(
                        "explicit candidate/material mapping requires a non-empty note".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StabilityEvidenceReceipt {
    pub schema: &'static str,
    pub capability_classification: &'static str,
    pub capture_sha256: String,
    pub raw_json_sha256: String,
    pub binding: StabilityBinding,
    pub material_id: String,
    pub formula_pretty: String,
    pub last_updated: String,
    pub source_warnings: Vec<String>,
    pub prediction: Prediction,
}

impl StabilityEvidenceReceipt {
    pub fn sha256(&self) -> Result<String, StabilityError> {
        let encoded = serde_json::to_vec(self)?;
        let mut hasher = Sha256::new();
        hasher.update(RECEIPT_DIGEST_DOMAIN);
        hasher.update(encoded);
        Ok(hex_lower(&hasher.finalize()))
    }
}

pub fn bind_stability_evidence(
    capture: &StabilityCapture,
    binding: StabilityBinding,
) -> Result<StabilityEvidenceReceipt, StabilityError> {
    capture.validate()?;
    binding.validate()?;
    let record = capture
        .records
        .iter()
        .find(|record| record.material_id == binding.material_id)
        .ok_or_else(|| StabilityError::MaterialNotFound(binding.material_id.clone()))?;
    if record.deprecated {
        return Err(StabilityError::DeprecatedMaterial(record.material_id.clone()));
    }

    let capture_sha256 = capture.sha256()?;
    let mut model = ModelProvenance::named("Materials Project thermodynamic SummaryDoc")?;
    model.version = Some(format!(
        "mp-api:{};emmet:{}",
        capture.metadata.mp_api_version, capture.metadata.emmet_version
    ));
    model.output_digest = Some(format!("sha256:{capture_sha256}"));

    let binding_assumption = match &binding.basis {
        BindingBasis::ExactMaterialId => {
            "Candidate identity exactly matches the Materials Project material_id.".to_owned()
        }
        BindingBasis::ExplicitMapping { note } => format!(
            "Candidate-to-Materials-Project identity is caller-declared, not independently proven: {note}"
        ),
    };

    let prediction = Prediction {
        metric: metric::ENERGY_ABOVE_HULL.to_owned(),
        value: record.energy_above_hull_ev_per_atom,
        unit: unit::EV_PER_ATOM.to_owned(),
        // SummaryDoc supplies no calibrated predictive interval for this
        // adapter. Epistemic=1.0 uses the generic contract's documented
        // 'unknown' endpoint; aleatoric=0.0 means this captured computed value
        // has no stochastic noise model attached, not that nature is noiseless.
        uncertainty: UncertaintyEstimate::new(1.0, 0.0)?,
        fidelity: FidelityLevel::FirstPrinciples,
        model,
        assumptions: vec![
            "Materials Project energy_above_hull is computed thermodynamic/phase-diagram evidence derived from first-principles calculations; it is not experimental stability evidence.".into(),
            "No calibrated predictive uncertainty interval is supplied by this adapter; normalized epistemic uncertainty is therefore marked fully unknown.".into(),
            "Aleatoric uncertainty is zero only because this database record carries no stochastic noise model; it is not a claim of experimental determinism.".into(),
            binding_assumption,
        ],
        evidence: vec![EvidenceRef {
            id: format!(
                "materials-project:{}:energy_above_hull",
                record.material_id
            ),
            kind: EvidenceKind::Dataset,
            uri: None,
            digest: Some(format!("sha256:{}", capture.raw_json_sha256)),
            note: Some(format!(
                "SummaryDoc capture; last_updated={}; source warnings={:?}",
                record.last_updated, record.warnings
            )),
        }],
    };
    prediction.validate()?;

    Ok(StabilityEvidenceReceipt {
        schema: "symthaea.materials-project.stability-evidence.v0",
        capability_classification: CAPABILITY_CLASSIFICATION,
        capture_sha256,
        raw_json_sha256: capture.raw_json_sha256.clone(),
        binding,
        material_id: record.material_id.clone(),
        formula_pretty: record.formula_pretty.clone(),
        last_updated: record.last_updated.clone(),
        source_warnings: record.warnings.clone(),
        prediction,
    })
}

#[derive(Debug, Error)]
pub enum StabilityError {
    #[error("invalid Materials Project capture metadata: {0}")]
    InvalidMetadata(String),
    #[error("invalid Materials Project stability capture: {0}")]
    InvalidCapture(String),
    #[error("invalid Materials Project stability record: {0}")]
    InvalidRecord(String),
    #[error("duplicate Materials Project material id {0:?}")]
    DuplicateMaterialId(String),
    #[error("invalid candidate/material binding: {0}")]
    InvalidBinding(String),
    #[error("Materials Project material {0:?} was not present in this capture")]
    MaterialNotFound(String),
    #[error("Materials Project material {0:?} is deprecated and cannot satisfy stability evidence")]
    DeprecatedMaterial(String),
    #[error(transparent)]
    Discovery(#[from] symthaea_discovery::DiscoveryError),
    #[error("Materials Project capture JSON failed to parse/serialize: {0}")]
    Json(#[from] serde_json::Error),
}

fn required_string(
    value: Option<String>,
    row_index: usize,
    field: &str,
) -> Result<String, StabilityError> {
    let value = value.ok_or_else(|| {
        StabilityError::InvalidRecord(format!("SummaryDoc row {row_index} is missing {field}"))
    })?;
    if value.trim().is_empty() {
        return Err(StabilityError::InvalidRecord(format!(
            "SummaryDoc row {row_index} has empty {field}"
        )));
    }
    Ok(value)
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

    fn metadata() -> CaptureMetadata {
        CaptureMetadata {
            source_name: "Materials Project".into(),
            api_route: SUMMARY_ROUTE.into(),
            mp_api_version: "fixture-mp-api".into(),
            emmet_version: "fixture-emmet".into(),
            retrieved_at_utc: "2026-09-12T00:00:00Z".into(),
            query_description: "fixture exact material ids".into(),
            requested_fields: REQUIRED_FIELDS.iter().map(|field| (*field).into()).collect(),
        }
    }

    fn raw_json() -> Vec<u8> {
        br#"[
          {
            "material_id":"mp-149",
            "formula_pretty":"Si",
            "energy_above_hull":0.0,
            "formation_energy_per_atom":0.0,
            "deprecated":false,
            "last_updated":"2026-08-01T00:00:00Z",
            "warnings":[],
            "extra_field_is_ignored_but_raw_digest_binds_it":"x"
          },
          {
            "material_id":"mp-2",
            "formula_pretty":"C",
            "energy_above_hull":0.02,
            "formation_energy_per_atom":-0.1,
            "deprecated":false,
            "last_updated":"2026-08-02T00:00:00Z",
            "warnings":["fixture warning"]
          }
        ]"#
            .to_vec()
    }

    #[test]
    fn raw_bytes_and_normalized_capture_are_both_content_addressed() {
        let first_raw = raw_json();
        let first = parse_summary_docs_json(&first_raw, metadata()).unwrap();
        let mut changed_raw = first_raw.clone();
        changed_raw.extend_from_slice(b" ");
        let changed = parse_summary_docs_json(&changed_raw, metadata()).unwrap();
        assert_ne!(first.raw_json_sha256, changed.raw_json_sha256);
        assert_ne!(first.sha256().unwrap(), changed.sha256().unwrap());
        assert_eq!(first.records[0].material_id, "mp-149");
        assert_eq!(first.records[1].material_id, "mp-2");
    }

    #[test]
    fn stability_prediction_is_first_principles_fidelity_but_dataset_evidence() {
        let capture = parse_summary_docs_json(&raw_json(), metadata()).unwrap();
        let receipt = bind_stability_evidence(
            &capture,
            StabilityBinding {
                candidate_id: CandidateId::new("mp-149").unwrap(),
                material_id: "mp-149".into(),
                basis: BindingBasis::ExactMaterialId,
            },
        )
        .unwrap();
        assert_eq!(receipt.prediction.metric, metric::ENERGY_ABOVE_HULL);
        assert_eq!(receipt.prediction.unit, unit::EV_PER_ATOM);
        assert_eq!(receipt.prediction.fidelity, FidelityLevel::FirstPrinciples);
        assert_eq!(receipt.prediction.evidence[0].kind, EvidenceKind::Dataset);
        assert_eq!(receipt.prediction.uncertainty.epistemic, 1.0);
        assert_eq!(receipt.prediction.uncertainty.aleatoric, 0.0);
        assert!(receipt.prediction.uncertainty.interval.is_none());
    }

    #[test]
    fn direct_material_binding_cannot_silently_map_a_different_candidate() {
        let capture = parse_summary_docs_json(&raw_json(), metadata()).unwrap();
        let error = bind_stability_evidence(
            &capture,
            StabilityBinding {
                candidate_id: CandidateId::new("generated-candidate").unwrap(),
                material_id: "mp-149".into(),
                basis: BindingBasis::ExactMaterialId,
            },
        )
        .unwrap_err();
        assert!(matches!(error, StabilityError::InvalidBinding(_)));
    }

    #[test]
    fn explicit_mapping_requires_a_reviewable_note() {
        let binding = StabilityBinding {
            candidate_id: CandidateId::new("generated-candidate").unwrap(),
            material_id: "mp-149".into(),
            basis: BindingBasis::ExplicitMapping { note: "   ".into() },
        };
        assert!(binding.validate().is_err());
    }

    #[test]
    fn deprecated_material_cannot_satisfy_stability_evidence() {
        let raw = br#"[{
          "material_id":"mp-old",
          "formula_pretty":"X",
          "energy_above_hull":0.0,
          "formation_energy_per_atom":null,
          "deprecated":true,
          "last_updated":"2026-08-01T00:00:00Z",
          "warnings":[]
        }]"#;
        let capture = parse_summary_docs_json(raw, metadata()).unwrap();
        let error = bind_stability_evidence(
            &capture,
            StabilityBinding {
                candidate_id: CandidateId::new("mp-old").unwrap(),
                material_id: "mp-old".into(),
                basis: BindingBasis::ExactMaterialId,
            },
        )
        .unwrap_err();
        assert!(matches!(error, StabilityError::DeprecatedMaterial(id) if id == "mp-old"));
    }

    #[test]
    fn duplicate_ids_and_negative_hull_energy_fail_closed() {
        let duplicate = br#"[
          {"material_id":"mp-x","formula_pretty":"X","energy_above_hull":0.0,"formation_energy_per_atom":null,"deprecated":false,"last_updated":"2026-01-01","warnings":[]},
          {"material_id":"mp-x","formula_pretty":"X","energy_above_hull":0.1,"formation_energy_per_atom":null,"deprecated":false,"last_updated":"2026-01-02","warnings":[]}
        ]"#;
        assert!(matches!(
            parse_summary_docs_json(duplicate, metadata()),
            Err(StabilityError::DuplicateMaterialId(id)) if id == "mp-x"
        ));

        let negative = br#"[{
          "material_id":"mp-negative","formula_pretty":"X","energy_above_hull":-0.001,
          "formation_energy_per_atom":null,"deprecated":false,"last_updated":"2026-01-01","warnings":[]
        }]"#;
        assert!(matches!(
            parse_summary_docs_json(negative, metadata()),
            Err(StabilityError::InvalidRecord(_))
        ));
    }

    #[test]
    fn capture_metadata_requires_all_evidence_fields() {
        let mut metadata = metadata();
        metadata.requested_fields.retain(|field| field != "energy_above_hull");
        assert!(metadata.validate().is_err());
    }
}
