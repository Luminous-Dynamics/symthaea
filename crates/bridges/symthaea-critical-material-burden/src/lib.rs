// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Content-addressed critical-material mass-fraction evidence.
//!
//! The changing definition of "critical" lives in an external, versioned list.
//! This crate only validates that list, parses a simple chemical formula, and
//! computes the transparent mass fraction contributed by designated elements.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use symthaea_bandgap::periodic_table::by_symbol;
use symthaea_discovery::{
    CandidateId, EvidenceKind, EvidenceRef, FidelityLevel, ModelProvenance, Prediction,
    UncertaintyEstimate,
};
use symthaea_energy_material_screening::{metric, unit};
use symthaea_process_discovery::formula::parse_formula;
use thiserror::Error;

pub const CAPABILITY_CLASSIFICATION: &str =
    "CRITICAL-MATERIAL BURDEN EVIDENCE ONLY -- designation-list mass fraction is not total supply risk, life-cycle impact, material safety, or deployment authority.";
const LIST_DIGEST_DOMAIN: &[u8] = b"symthaea.critical-material.designation-list.v0\0";
const RECEIPT_DIGEST_DOMAIN: &[u8] = b"symthaea.critical-material.burden-receipt.v0\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CriticalElementList {
    pub source_title: String,
    pub source_version: String,
    pub source_uri: String,
    pub source_document_sha256: String,
    pub normalized_capture_sha256: String,
    pub normalized_at_utc: String,
    pub extraction_note: String,
    pub elements: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawCriticalElementList {
    source_title: String,
    source_version: String,
    source_uri: String,
    source_document_sha256: String,
    normalized_at_utc: String,
    extraction_note: String,
    elements: Vec<String>,
}

impl CriticalElementList {
    pub fn from_json_bytes(raw_json: &[u8]) -> Result<Self, CriticalBurdenError> {
        let raw: RawCriticalElementList = serde_json::from_slice(raw_json)?;
        let mut elements = raw.elements;
        elements.sort();
        let value = Self {
            source_title: raw.source_title,
            source_version: raw.source_version,
            source_uri: raw.source_uri,
            source_document_sha256: raw.source_document_sha256,
            normalized_capture_sha256: sha256_bytes(raw_json),
            normalized_at_utc: raw.normalized_at_utc,
            extraction_note: raw.extraction_note,
            elements,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), CriticalBurdenError> {
        for (name, value) in [
            ("source_title", self.source_title.as_str()),
            ("source_version", self.source_version.as_str()),
            ("source_uri", self.source_uri.as_str()),
            ("normalized_at_utc", self.normalized_at_utc.as_str()),
            ("extraction_note", self.extraction_note.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(CriticalBurdenError::InvalidList(format!(
                    "{name} cannot be empty"
                )));
            }
        }
        for (name, digest) in [
            ("source_document_sha256", self.source_document_sha256.as_str()),
            (
                "normalized_capture_sha256",
                self.normalized_capture_sha256.as_str(),
            ),
        ] {
            if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(CriticalBurdenError::InvalidList(format!(
                    "{name} must be 64 hexadecimal characters"
                )));
            }
        }
        if self.elements.is_empty() {
            return Err(CriticalBurdenError::InvalidList(
                "critical-element list cannot be empty".into(),
            ));
        }
        let mut seen = BTreeSet::new();
        for symbol in &self.elements {
            if by_symbol(symbol).is_none() {
                return Err(CriticalBurdenError::UnknownElement(symbol.clone()));
            }
            if !seen.insert(symbol.as_str()) {
                return Err(CriticalBurdenError::InvalidList(format!(
                    "duplicate critical element {symbol:?}"
                )));
            }
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String, CriticalBurdenError> {
        self.validate()?;
        let mut canonical = self.clone();
        canonical.elements.sort();
        let encoded = serde_json::to_vec(&canonical)?;
        let mut hasher = Sha256::new();
        hasher.update(LIST_DIGEST_DOMAIN);
        hasher.update(encoded);
        Ok(hex_lower(&hasher.finalize()))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElementMassContribution {
    pub symbol: String,
    pub atom_count: u32,
    pub atomic_mass_amu: f64,
    pub formula_mass_amu: f64,
    pub mass_fraction: f64,
    pub designated_critical: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CriticalBurdenReceipt {
    pub schema: String,
    pub capability_classification: String,
    pub candidate_id: CandidateId,
    pub formula: String,
    pub designation_list_sha256: String,
    pub source_document_sha256: String,
    pub normalized_capture_sha256: String,
    pub contributions: Vec<ElementMassContribution>,
    pub critical_mass_fraction: f64,
    pub prediction: Prediction,
}

impl CriticalBurdenReceipt {
    pub fn sha256(&self) -> Result<String, CriticalBurdenError> {
        let encoded = serde_json::to_vec(self)?;
        let mut hasher = Sha256::new();
        hasher.update(RECEIPT_DIGEST_DOMAIN);
        hasher.update(encoded);
        Ok(hex_lower(&hasher.finalize()))
    }
}

pub fn calculate_critical_material_burden(
    candidate_id: CandidateId,
    formula: &str,
    list: &CriticalElementList,
) -> Result<CriticalBurdenReceipt, CriticalBurdenError> {
    CandidateId::new(candidate_id.0.clone())?;
    list.validate()?;
    if formula.trim().is_empty() {
        return Err(CriticalBurdenError::InvalidFormula(
            "formula cannot be empty".into(),
        ));
    }
    let parsed = parse_formula(formula).ok_or_else(|| {
        CriticalBurdenError::InvalidFormula(format!(
            "formula {formula:?} is outside the simple Hill-shaped parser scope"
        ))
    })?;
    let critical: BTreeSet<&str> = list.elements.iter().map(String::as_str).collect();

    let mut raw = Vec::with_capacity(parsed.len());
    let mut total_mass = 0.0f64;
    for (symbol, atom_count) in parsed {
        let element = by_symbol(&symbol)
            .ok_or_else(|| CriticalBurdenError::UnknownElement(symbol.clone()))?;
        if !element.atomic_mass.is_finite() || element.atomic_mass <= 0.0 {
            return Err(CriticalBurdenError::InvalidAtomicMass(symbol));
        }
        let formula_mass = element.atomic_mass * f64::from(atom_count);
        total_mass += formula_mass;
        raw.push((
            element.symbol.to_owned(),
            atom_count,
            element.atomic_mass,
            formula_mass,
            critical.contains(element.symbol),
        ));
    }
    if !total_mass.is_finite() || total_mass <= 0.0 {
        return Err(CriticalBurdenError::InvalidFormula(
            "formula produced non-positive total mass".into(),
        ));
    }

    let mut critical_mass = 0.0f64;
    let contributions: Vec<ElementMassContribution> = raw
        .into_iter()
        .map(|(symbol, atom_count, atomic_mass_amu, formula_mass_amu, designated_critical)| {
            if designated_critical {
                critical_mass += formula_mass_amu;
            }
            ElementMassContribution {
                symbol,
                atom_count,
                atomic_mass_amu,
                formula_mass_amu,
                mass_fraction: formula_mass_amu / total_mass,
                designated_critical,
            }
        })
        .collect();
    let critical_mass_fraction = critical_mass / total_mass;
    if !(0.0..=1.0).contains(&critical_mass_fraction) || !critical_mass_fraction.is_finite() {
        return Err(CriticalBurdenError::InvalidFormula(
            "computed critical mass fraction fell outside [0,1]".into(),
        ));
    }

    let list_sha256 = list.sha256()?;
    let mut model = ModelProvenance::named("stoichiometric critical-material mass fraction")?;
    model.version = Some("v0".into());
    model.input_digest = Some(format!("sha256:{list_sha256}"));

    let prediction = Prediction {
        metric: metric::CRITICAL_MATERIAL_MASS_FRACTION.to_owned(),
        value: critical_mass_fraction,
        unit: unit::FRACTION.to_owned(),
        uncertainty: UncertaintyEstimate::new(1.0, 0.0)?,
        fidelity: FidelityLevel::Analytical,
        model,
        assumptions: vec![
            "Criticality is defined exclusively by the exact externally supplied designation list; this adapter does not define which elements should be considered critical.".into(),
            "The formula is treated as exact stoichiometry and mass fraction uses the repository's standard atomic masses.".into(),
            "A zero critical mass fraction means no formula element appears on this exact designation list; it is not a claim of zero supply, geopolitical, environmental, or economic risk.".into(),
            "The simple formula parser does not support parentheses, hydrates, fractional occupancy, disorder, isotopic notation, or non-stoichiometry; unsupported formulas fail closed.".into(),
            "No calibrated uncertainty model is available for designation-list applicability; generic epistemic uncertainty is marked fully unknown.".into(),
        ],
        evidence: vec![EvidenceRef {
            id: format!("critical-element-list:{}", list.source_version),
            kind: EvidenceKind::Dataset,
            uri: Some(list.source_uri.clone()),
            digest: Some(format!("sha256:{}", list.source_document_sha256)),
            note: Some(format!(
                "normalized capture sha256={}; extraction note={}",
                list.normalized_capture_sha256, list.extraction_note
            )),
        }],
    };
    prediction.validate()?;

    Ok(CriticalBurdenReceipt {
        schema: "symthaea.critical-material.burden-receipt.v0".into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        candidate_id,
        formula: formula.to_owned(),
        designation_list_sha256: list_sha256,
        source_document_sha256: list.source_document_sha256.clone(),
        normalized_capture_sha256: list.normalized_capture_sha256.clone(),
        contributions,
        critical_mass_fraction,
        prediction,
    })
}

#[derive(Debug, Error)]
pub enum CriticalBurdenError {
    #[error("invalid critical-element list: {0}")]
    InvalidList(String),
    #[error("invalid candidate formula: {0}")]
    InvalidFormula(String),
    #[error("unknown/unsupported element symbol {0:?}")]
    UnknownElement(String),
    #[error("element {0:?} has invalid atomic mass")]
    InvalidAtomicMass(String),
    #[error(transparent)]
    Discovery(#[from] symthaea_discovery::DiscoveryError),
    #[error("critical-material evidence JSON failed to parse/serialize: {0}")]
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

    fn list_json(elements: &[&str]) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "source_title": "Fixture Critical Elements",
            "source_version": "2026-fixture",
            "source_uri": "https://example.invalid/critical-list",
            "source_document_sha256": "1".repeat(64),
            "normalized_at_utc": "2026-09-12T00:00:00Z",
            "extraction_note": "test fixture only",
            "elements": elements,
        }))
        .unwrap()
    }

    #[test]
    fn lifepo4_lithium_mass_fraction_is_transparent() {
        let list = CriticalElementList::from_json_bytes(&list_json(&["Li"])).unwrap();
        let receipt = calculate_critical_material_burden(
            CandidateId::new("LiFePO4").unwrap(),
            "LiFePO4",
            &list,
        )
        .unwrap();

        let li = by_symbol("Li").unwrap().atomic_mass;
        let fe = by_symbol("Fe").unwrap().atomic_mass;
        let p = by_symbol("P").unwrap().atomic_mass;
        let o = by_symbol("O").unwrap().atomic_mass;
        let expected = li / (li + fe + p + 4.0 * o);
        assert!((receipt.critical_mass_fraction - expected).abs() < 1e-12);
        assert_eq!(receipt.prediction.metric, metric::CRITICAL_MATERIAL_MASS_FRACTION);
        assert_eq!(receipt.prediction.fidelity, FidelityLevel::Analytical);
        assert_eq!(receipt.prediction.evidence[0].kind, EvidenceKind::Dataset);
    }

    #[test]
    fn zero_is_valid_only_relative_to_the_exact_designation_list() {
        let list = CriticalElementList::from_json_bytes(&list_json(&["Li"])).unwrap();
        let receipt = calculate_critical_material_burden(
            CandidateId::new("FePO4").unwrap(),
            "FePO4",
            &list,
        )
        .unwrap();
        assert_eq!(receipt.critical_mass_fraction, 0.0);
        assert!(receipt
            .prediction
            .assumptions
            .iter()
            .any(|value| value.contains("not a claim of zero supply")));
    }

    #[test]
    fn designation_list_order_does_not_change_semantic_identity() {
        let a = CriticalElementList::from_json_bytes(&list_json(&["Li", "Co"])).unwrap();
        let b = CriticalElementList::from_json_bytes(&list_json(&["Co", "Li"])).unwrap();
        // Raw normalized-capture identity differs, but semantic list identity is
        // canonicalized except for that explicitly retained source-capture hash.
        assert_ne!(a.normalized_capture_sha256, b.normalized_capture_sha256);
        assert_ne!(a.sha256().unwrap(), b.sha256().unwrap());
        assert_eq!(a.elements, b.elements);
    }

    #[test]
    fn duplicate_or_unknown_designations_fail_closed() {
        assert!(CriticalElementList::from_json_bytes(&list_json(&["Li", "Li"])).is_err());
        assert!(CriticalElementList::from_json_bytes(&list_json(&["Xx"])).is_err());
    }

    #[test]
    fn unsupported_formula_syntax_fails_closed() {
        let list = CriticalElementList::from_json_bytes(&list_json(&["Li"])).unwrap();
        assert!(calculate_critical_material_burden(
            CandidateId::new("candidate").unwrap(),
            "Ca(OH)2",
            &list,
        )
        .is_err());
    }
}
