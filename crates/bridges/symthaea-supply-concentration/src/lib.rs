// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Content-addressed supply-concentration evidence for energy-material screening.
//!
//! Supply-chain stage, temporal basis, source bytes, elemental producer shares,
//! and candidate aggregation policy are all explicit. Missing elemental supply
//! data fails closed instead of being silently ignored.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_bandgap::periodic_table::by_symbol;
use symthaea_discovery::{
    CandidateId, EvidenceKind, EvidenceRef, FidelityLevel, ModelProvenance, Prediction,
    UncertaintyEstimate,
};
use symthaea_energy_material_screening::{metric, unit};
use symthaea_process_discovery::formula::parse_formula;
use thiserror::Error;

pub const CAPABILITY_CLASSIFICATION: &str =
    "SUPPLY-CONCENTRATION EVIDENCE ONLY -- concentration is not total supply resilience, criticality, scarcity, substitutability, geopolitical risk, or deployment authority.";
const DATASET_DIGEST_DOMAIN: &[u8] = b"symthaea.supply-concentration.dataset.v0\0";
const RECEIPT_DIGEST_DOMAIN: &[u8] = b"symthaea.supply-concentration.receipt.v0\0";
const SHARE_SUM_TOLERANCE: f64 = 1e-6;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SupplyStage {
    Mining,
    Refining,
    Processing,
    Manufacturing,
    Other { name: String },
}

impl SupplyStage {
    fn validate(&self) -> Result<(), SupplyError> {
        if let Self::Other { name } = self {
            if name.trim().is_empty() {
                return Err(SupplyError::InvalidDataset(
                    "custom supply stage cannot be blank".into(),
                ));
            }
        }
        Ok(())
    }

    fn label(&self) -> String {
        match self {
            Self::Mining => "mining".into(),
            Self::Refining => "refining".into(),
            Self::Processing => "processing".into(),
            Self::Manufacturing => "manufacturing".into(),
            Self::Other { name } => format!("other:{name}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SupplyDataBasis {
    Observed { year: u16 },
    Estimated { year: u16 },
    Projected { year: u16, scenario: String },
}

impl SupplyDataBasis {
    fn validate(&self) -> Result<(), SupplyError> {
        let year = match self {
            Self::Observed { year } | Self::Estimated { year } => *year,
            Self::Projected { year, scenario } => {
                if scenario.trim().is_empty() {
                    return Err(SupplyError::InvalidDataset(
                        "projected supply data requires a non-empty scenario".into(),
                    ));
                }
                *year
            }
        };
        if !(1900..=2200).contains(&year) {
            return Err(SupplyError::InvalidDataset(format!(
                "supply-data year {year} is outside supported review bounds"
            )));
        }
        Ok(())
    }

    fn label(&self) -> String {
        match self {
            Self::Observed { year } => format!("observed:{year}"),
            Self::Estimated { year } => format!("estimated:{year}"),
            Self::Projected { year, scenario } => format!("projected:{year}:{scenario}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProducerShare {
    pub jurisdiction: String,
    /// Fraction of the normalized global/declared market, in [0,1].
    pub share: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElementSupplyRecord {
    pub symbol: String,
    pub producers: Vec<ProducerShare>,
}

impl ElementSupplyRecord {
    fn validate(&self) -> Result<(), SupplyError> {
        if by_symbol(&self.symbol).is_none() {
            return Err(SupplyError::UnknownElement(self.symbol.clone()));
        }
        if self.producers.is_empty() {
            return Err(SupplyError::InvalidDataset(format!(
                "{} has no producer shares",
                self.symbol
            )));
        }
        let mut jurisdictions = BTreeSet::new();
        let mut total = 0.0;
        for producer in &self.producers {
            if producer.jurisdiction.trim().is_empty() {
                return Err(SupplyError::InvalidDataset(format!(
                    "{} contains a blank producer jurisdiction",
                    self.symbol
                )));
            }
            if !jurisdictions.insert(producer.jurisdiction.as_str()) {
                return Err(SupplyError::InvalidDataset(format!(
                    "{} repeats producer jurisdiction {:?}",
                    self.symbol, producer.jurisdiction
                )));
            }
            if !producer.share.is_finite() || !(0.0..=1.0).contains(&producer.share) {
                return Err(SupplyError::InvalidDataset(format!(
                    "{} producer {:?} has invalid share {}",
                    self.symbol, producer.jurisdiction, producer.share
                )));
            }
            total += producer.share;
        }
        if (total - 1.0).abs() > SHARE_SUM_TOLERANCE {
            return Err(SupplyError::InvalidDataset(format!(
                "{} producer shares must sum to 1 within tolerance; got {total}",
                self.symbol
            )));
        }
        Ok(())
    }

    pub fn normalized_hhi(&self) -> Result<f64, SupplyError> {
        self.validate()?;
        let hhi: f64 = self.producers.iter().map(|producer| producer.share.powi(2)).sum();
        if !hhi.is_finite() || !(0.0..=1.0 + SHARE_SUM_TOLERANCE).contains(&hhi) {
            return Err(SupplyError::InvalidDataset(format!(
                "{} produced invalid normalized HHI {hhi}",
                self.symbol
            )));
        }
        Ok(hhi.clamp(0.0, 1.0))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SupplyDataset {
    pub source_title: String,
    pub source_version: String,
    pub source_uri: String,
    pub source_document_sha256: String,
    pub normalized_capture_sha256: String,
    pub normalized_at_utc: String,
    pub extraction_note: String,
    pub stage: SupplyStage,
    pub basis: SupplyDataBasis,
    pub records: Vec<ElementSupplyRecord>,
}

#[derive(Debug, Deserialize)]
struct RawSupplyDataset {
    source_title: String,
    source_version: String,
    source_uri: String,
    source_document_sha256: String,
    normalized_at_utc: String,
    extraction_note: String,
    stage: SupplyStage,
    basis: SupplyDataBasis,
    records: Vec<ElementSupplyRecord>,
}

impl SupplyDataset {
    pub fn from_json_bytes(raw_json: &[u8]) -> Result<Self, SupplyError> {
        let raw: RawSupplyDataset = serde_json::from_slice(raw_json)?;
        let mut records = raw.records;
        records.sort_by(|left, right| left.symbol.cmp(&right.symbol));
        for record in &mut records {
            record
                .producers
                .sort_by(|left, right| left.jurisdiction.cmp(&right.jurisdiction));
        }
        let dataset = Self {
            source_title: raw.source_title,
            source_version: raw.source_version,
            source_uri: raw.source_uri,
            source_document_sha256: raw.source_document_sha256,
            normalized_capture_sha256: sha256_bytes(raw_json),
            normalized_at_utc: raw.normalized_at_utc,
            extraction_note: raw.extraction_note,
            stage: raw.stage,
            basis: raw.basis,
            records,
        };
        dataset.validate()?;
        Ok(dataset)
    }

    pub fn validate(&self) -> Result<(), SupplyError> {
        for (name, value) in [
            ("source_title", self.source_title.as_str()),
            ("source_version", self.source_version.as_str()),
            ("source_uri", self.source_uri.as_str()),
            ("normalized_at_utc", self.normalized_at_utc.as_str()),
            ("extraction_note", self.extraction_note.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(SupplyError::InvalidDataset(format!("{name} cannot be blank")));
            }
        }
        for (name, digest) in [
            ("source_document_sha256", self.source_document_sha256.as_str()),
            ("normalized_capture_sha256", self.normalized_capture_sha256.as_str()),
        ] {
            if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(SupplyError::InvalidDataset(format!(
                    "{name} must be 64 hexadecimal characters"
                )));
            }
        }
        self.stage.validate()?;
        self.basis.validate()?;
        if self.records.is_empty() {
            return Err(SupplyError::InvalidDataset(
                "supply dataset must contain at least one element".into(),
            ));
        }
        let mut symbols = BTreeSet::new();
        for record in &self.records {
            record.validate()?;
            if !symbols.insert(record.symbol.as_str()) {
                return Err(SupplyError::InvalidDataset(format!(
                    "duplicate elemental supply record {:?}",
                    record.symbol
                )));
            }
        }
        Ok(())
    }

    pub fn verify_source_document(&self, source_document: &[u8]) -> Result<(), SupplyError> {
        self.validate()?;
        let actual = sha256_bytes(source_document);
        if !actual.eq_ignore_ascii_case(&self.source_document_sha256) {
            return Err(SupplyError::SourceDocumentDigestMismatch {
                expected: self.source_document_sha256.clone(),
                actual,
            });
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, SupplyError> {
        self.validate()?;
        let mut canonical = self.clone();
        canonical.records.sort_by(|left, right| left.symbol.cmp(&right.symbol));
        for record in &mut canonical.records {
            record
                .producers
                .sort_by(|left, right| left.jurisdiction.cmp(&right.jurisdiction));
        }
        let bytes = serde_json::to_vec(&canonical)?;
        let mut hasher = Sha256::new();
        hasher.update(DATASET_DIGEST_DOMAIN);
        hasher.update(bytes);
        Ok(hex_lower(&hasher.finalize()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateAggregationPolicy {
    /// Candidate inherits the highest elemental concentration value. Useful as a
    /// conservative bottleneck screen; it intentionally ignores mass fraction.
    MaximumElementHhi,
    /// Candidate value is the formula-mass-weighted mean elemental HHI.
    MassWeightedMeanHhi,
}

impl CandidateAggregationPolicy {
    fn label(self) -> &'static str {
        match self {
            Self::MaximumElementHhi => "maximum_element_hhi",
            Self::MassWeightedMeanHhi => "mass_weighted_mean_hhi",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElementSupplyContribution {
    pub symbol: String,
    pub atom_count: u32,
    pub atomic_mass_amu: f64,
    pub formula_mass_fraction: f64,
    pub normalized_hhi: f64,
    pub conventional_hhi_points: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SupplyConcentrationReceipt {
    pub schema: String,
    pub capability_classification: String,
    pub candidate_id: CandidateId,
    pub formula: String,
    pub stage: SupplyStage,
    pub basis: SupplyDataBasis,
    pub aggregation_policy: CandidateAggregationPolicy,
    pub source_document_sha256: String,
    pub normalized_capture_sha256: String,
    pub supply_dataset_sha256: String,
    pub contributions: Vec<ElementSupplyContribution>,
    /// HHI using fractional shares, in [0,1]. Multiply by 10,000 to obtain the
    /// conventional percentage-share HHI point scale.
    pub candidate_normalized_hhi: f64,
    pub prediction: Prediction,
}

impl SupplyConcentrationReceipt {
    pub fn sha256(&self) -> Result<String, SupplyError> {
        let bytes = serde_json::to_vec(self)?;
        let mut hasher = Sha256::new();
        hasher.update(RECEIPT_DIGEST_DOMAIN);
        hasher.update(bytes);
        Ok(hex_lower(&hasher.finalize()))
    }
}

pub fn calculate_supply_concentration(
    candidate_id: CandidateId,
    formula: &str,
    dataset: &SupplyDataset,
    source_document: &[u8],
    aggregation_policy: CandidateAggregationPolicy,
) -> Result<SupplyConcentrationReceipt, SupplyError> {
    CandidateId::new(candidate_id.0.clone())?;
    dataset.verify_source_document(source_document)?;
    let parsed = parse_formula(formula).ok_or_else(|| {
        SupplyError::InvalidFormula(format!(
            "formula {formula:?} is outside the simple Hill-shaped parser scope"
        ))
    })?;
    if parsed.is_empty() {
        return Err(SupplyError::InvalidFormula("formula cannot be empty".into()));
    }

    let records: BTreeMap<&str, &ElementSupplyRecord> = dataset
        .records
        .iter()
        .map(|record| (record.symbol.as_str(), record))
        .collect();

    let mut raw = Vec::with_capacity(parsed.len());
    let mut total_mass = 0.0;
    for (symbol, atom_count) in parsed {
        if atom_count == 0 {
            return Err(SupplyError::InvalidFormula(format!(
                "element {symbol:?} has zero stoichiometric count"
            )));
        }
        let element = by_symbol(&symbol).ok_or_else(|| SupplyError::UnknownElement(symbol.clone()))?;
        if !element.atomic_mass.is_finite() || element.atomic_mass <= 0.0 {
            return Err(SupplyError::InvalidFormula(format!(
                "element {symbol:?} has invalid standard atomic mass"
            )));
        }
        let record = records
            .get(symbol.as_str())
            .copied()
            .ok_or_else(|| SupplyError::MissingElementSupplyData(symbol.clone()))?;
        let formula_mass = element.atomic_mass * f64::from(atom_count);
        total_mass += formula_mass;
        raw.push((
            symbol,
            atom_count,
            element.atomic_mass,
            formula_mass,
            record.normalized_hhi()?,
        ));
    }
    if !total_mass.is_finite() || total_mass <= 0.0 {
        return Err(SupplyError::InvalidFormula(
            "formula produced invalid total mass".into(),
        ));
    }

    let contributions: Vec<ElementSupplyContribution> = raw
        .into_iter()
        .map(|(symbol, atom_count, atomic_mass_amu, formula_mass, normalized_hhi)| {
            ElementSupplyContribution {
                symbol,
                atom_count,
                atomic_mass_amu,
                formula_mass_fraction: formula_mass / total_mass,
                normalized_hhi,
                conventional_hhi_points: normalized_hhi * 10_000.0,
            }
        })
        .collect();

    let candidate_normalized_hhi = match aggregation_policy {
        CandidateAggregationPolicy::MaximumElementHhi => contributions
            .iter()
            .map(|entry| entry.normalized_hhi)
            .fold(0.0_f64, f64::max),
        CandidateAggregationPolicy::MassWeightedMeanHhi => contributions
            .iter()
            .map(|entry| entry.formula_mass_fraction * entry.normalized_hhi)
            .sum(),
    };
    if !candidate_normalized_hhi.is_finite() || !(0.0..=1.0).contains(&candidate_normalized_hhi) {
        return Err(SupplyError::InvalidDataset(format!(
            "candidate aggregation produced invalid normalized HHI {candidate_normalized_hhi}"
        )));
    }

    let dataset_sha256 = dataset.sha256()?;
    let mut model = ModelProvenance::named("elemental producer-share HHI aggregation")?;
    model.version = Some(format!(
        "v0;stage={};basis={};aggregation={}",
        dataset.stage.label(),
        dataset.basis.label(),
        aggregation_policy.label()
    ));
    model.input_digest = Some(format!("sha256:{dataset_sha256}"));

    let prediction = Prediction {
        metric: metric::SUPPLY_CONCENTRATION_HHI.to_owned(),
        value: candidate_normalized_hhi,
        unit: unit::SCORE.to_owned(),
        uncertainty: UncertaintyEstimate::new(1.0, 0.0)?,
        fidelity: FidelityLevel::Analytical,
        model,
        assumptions: vec![
            "Elemental concentration uses HHI = sum(share_i^2) on normalized producer/jurisdiction shares. This adapter reports HHI on a 0..1 scale; multiplying by 10,000 yields the conventional percentage-share HHI point scale.".into(),
            format!("Supply-chain stage is explicitly {:?}; mining, refining, processing, and manufacturing concentration are not interchangeable.", dataset.stage),
            format!("Supply data temporal basis is {:?}.", dataset.basis),
            format!("Candidate aggregation policy is {:?}; a different aggregation policy is a different method and must not be silently substituted.", aggregation_policy),
            "All elements present in the candidate formula must have supply-share records. Missing elements fail closed instead of being ignored.".into(),
            "This metric measures concentration only. It does not include absolute production volume, reserves, substitution, recycling, trade barriers, political stability, co-product dependence, demand growth, or disruption probability.".into(),
            "No calibrated uncertainty model is available for the normalized source table or candidate aggregation; epistemic uncertainty is marked fully unknown.".into(),
        ],
        evidence: vec![EvidenceRef {
            id: format!("supply-concentration:{}:{}", dataset.source_version, dataset.stage.label()),
            kind: EvidenceKind::Dataset,
            uri: Some(dataset.source_uri.clone()),
            digest: Some(format!("sha256:{}", dataset.source_document_sha256)),
            note: Some(format!(
                "normalized capture sha256={}; extraction note={}",
                dataset.normalized_capture_sha256, dataset.extraction_note
            )),
        }],
    };
    prediction.validate()?;

    Ok(SupplyConcentrationReceipt {
        schema: "symthaea.supply-concentration.receipt.v0".into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        candidate_id,
        formula: formula.to_owned(),
        stage: dataset.stage.clone(),
        basis: dataset.basis.clone(),
        aggregation_policy,
        source_document_sha256: dataset.source_document_sha256.clone(),
        normalized_capture_sha256: dataset.normalized_capture_sha256.clone(),
        supply_dataset_sha256: dataset_sha256,
        contributions,
        candidate_normalized_hhi,
        prediction,
    })
}

#[derive(Debug, Error)]
pub enum SupplyError {
    #[error("invalid supply-concentration dataset: {0}")]
    InvalidDataset(String),
    #[error("source document SHA-256 mismatch: expected {expected}, got {actual}")]
    SourceDocumentDigestMismatch { expected: String, actual: String },
    #[error("invalid candidate formula: {0}")]
    InvalidFormula(String),
    #[error("unknown/unsupported element symbol {0:?}")]
    UnknownElement(String),
    #[error("candidate formula contains {0:?} but the supply dataset has no record for it")]
    MissingElementSupplyData(String),
    #[error(transparent)]
    Discovery(#[from] symthaea_discovery::DiscoveryError),
    #[error("supply-concentration JSON failed to parse/serialize: {0}")]
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

    const SOURCE: &[u8] = b"fixture supply source";

    fn dataset_json(stage: SupplyStage) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "source_title": "Fixture Elemental Supply Shares",
            "source_version": "2026-fixture",
            "source_uri": "https://example.invalid/supply",
            "source_document_sha256": sha256_bytes(SOURCE),
            "normalized_at_utc": "2026-09-12T00:00:00Z",
            "extraction_note": "test fixture only; shares include Other so each element sums to one",
            "stage": stage,
            "basis": { "observed": { "year": 2025 } },
            "records": [
                {
                    "symbol": "Li",
                    "producers": [
                        {"jurisdiction": "A", "share": 0.5},
                        {"jurisdiction": "B", "share": 0.3},
                        {"jurisdiction": "Other", "share": 0.2}
                    ]
                },
                {
                    "symbol": "Fe",
                    "producers": [
                        {"jurisdiction": "A", "share": 0.25},
                        {"jurisdiction": "B", "share": 0.25},
                        {"jurisdiction": "C", "share": 0.25},
                        {"jurisdiction": "Other", "share": 0.25}
                    ]
                },
                {
                    "symbol": "P",
                    "producers": [
                        {"jurisdiction": "A", "share": 0.4},
                        {"jurisdiction": "B", "share": 0.3},
                        {"jurisdiction": "Other", "share": 0.3}
                    ]
                },
                {
                    "symbol": "O",
                    "producers": [
                        {"jurisdiction": "A", "share": 0.5},
                        {"jurisdiction": "B", "share": 0.5}
                    ]
                }
            ]
        }))
        .unwrap()
    }

    #[test]
    fn elemental_hhi_matches_sum_of_squared_fractional_shares() {
        let dataset = SupplyDataset::from_json_bytes(&dataset_json(SupplyStage::Mining)).unwrap();
        let lithium = dataset.records.iter().find(|record| record.symbol == "Li").unwrap();
        assert!((lithium.normalized_hhi().unwrap() - 0.38).abs() < 1e-12);
    }

    #[test]
    fn max_and_mass_weighted_policies_are_explicitly_different() {
        let dataset = SupplyDataset::from_json_bytes(&dataset_json(SupplyStage::Mining)).unwrap();
        let candidate = CandidateId::new("LiFePO4").unwrap();
        let max = calculate_supply_concentration(
            candidate.clone(),
            "LiFePO4",
            &dataset,
            SOURCE,
            CandidateAggregationPolicy::MaximumElementHhi,
        )
        .unwrap();
        let weighted = calculate_supply_concentration(
            candidate,
            "LiFePO4",
            &dataset,
            SOURCE,
            CandidateAggregationPolicy::MassWeightedMeanHhi,
        )
        .unwrap();
        assert_ne!(max.candidate_normalized_hhi, weighted.candidate_normalized_hhi);
        assert_ne!(max.prediction.model.version, weighted.prediction.model.version);
    }

    #[test]
    fn missing_element_supply_data_fails_closed() {
        let mut raw: serde_json::Value = serde_json::from_slice(&dataset_json(SupplyStage::Mining)).unwrap();
        raw["records"].as_array_mut().unwrap().retain(|record| record["symbol"] != "O");
        let bytes = serde_json::to_vec(&raw).unwrap();
        let dataset = SupplyDataset::from_json_bytes(&bytes).unwrap();
        assert!(matches!(
            calculate_supply_concentration(
                CandidateId::new("LiFePO4").unwrap(),
                "LiFePO4",
                &dataset,
                SOURCE,
                CandidateAggregationPolicy::MaximumElementHhi,
            ),
            Err(SupplyError::MissingElementSupplyData(symbol)) if symbol == "O"
        ));
    }

    #[test]
    fn source_bytes_and_share_completeness_are_enforced() {
        let dataset = SupplyDataset::from_json_bytes(&dataset_json(SupplyStage::Mining)).unwrap();
        assert!(matches!(
            calculate_supply_concentration(
                CandidateId::new("LiFePO4").unwrap(),
                "LiFePO4",
                &dataset,
                b"wrong source",
                CandidateAggregationPolicy::MaximumElementHhi,
            ),
            Err(SupplyError::SourceDocumentDigestMismatch { .. })
        ));

        let mut raw: serde_json::Value = serde_json::from_slice(&dataset_json(SupplyStage::Mining)).unwrap();
        raw["records"][0]["producers"][0]["share"] = serde_json::json!(0.4);
        let invalid = serde_json::to_vec(&raw).unwrap();
        assert!(SupplyDataset::from_json_bytes(&invalid).is_err());
    }

    #[test]
    fn stage_identity_is_not_interchangeable() {
        let mining = SupplyDataset::from_json_bytes(&dataset_json(SupplyStage::Mining)).unwrap();
        let refining = SupplyDataset::from_json_bytes(&dataset_json(SupplyStage::Refining)).unwrap();
        assert_ne!(mining.sha256().unwrap(), refining.sha256().unwrap());
    }
}
