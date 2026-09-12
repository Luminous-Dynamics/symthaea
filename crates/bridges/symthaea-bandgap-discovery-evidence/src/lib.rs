// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Composition-only band-gap prediction adapted into generic discovery evidence.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use symthaea_bandgap::{
    ml_bandgap::CompositionBandgapPredictor,
    periodic_table::element,
    training_data::load_training_data,
};
use symthaea_discovery::{
    CandidateId, EvidenceKind, EvidenceRef, FidelityLevel, ModelProvenance, Prediction,
    UncertaintyEstimate,
};
use symthaea_energy_material_screening::{metric, unit};
use thiserror::Error;

pub const CAPABILITY_CLASSIFICATION: &str =
    "COMPOSITION-ONLY SURROGATE BAND-GAP EVIDENCE -- not first-principles simulation, experimental validation, calibrated uncertainty, novelty, or deployment authority.";
pub const MODEL_SOURCE_BLOB_SHA1: &str = "9f5d0e5dc81e346c7558dca0a29e9e05925d88f9";
pub const TRAINING_SOURCE_BLOB_SHA1: &str = "59a001b2747317801857e6df3b66247514645690";
const TRAINING_DIGEST_DOMAIN: &[u8] = b"symthaea.bandgap.curated-training-table.v0\0";
const RECEIPT_DIGEST_DOMAIN: &[u8] = b"symthaea.bandgap.discovery-evidence.v0\0";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BandgapEvidenceReceipt {
    pub schema: String,
    pub capability_classification: String,
    pub candidate_id: CandidateId,
    pub composition: Vec<(u8, f64)>,
    pub training_table_sha256: String,
    pub training_entry_count: usize,
    pub model_source_blob_sha1: String,
    pub training_source_blob_sha1: String,
    pub baseline_gap_ev: f64,
    pub ml_correction_ev: f64,
    /// Diagnostic inter-tree standard deviation from the RF. This is retained
    /// for review only and is deliberately not promoted into calibrated
    /// `Prediction::uncertainty`.
    pub inter_tree_stddev_ev: f64,
    pub prediction: Prediction,
}

impl BandgapEvidenceReceipt {
    pub fn sha256(&self) -> Result<String, BandgapEvidenceError> {
        let bytes = serde_json::to_vec(self)?;
        let mut hasher = Sha256::new();
        hasher.update(RECEIPT_DIGEST_DOMAIN);
        hasher.update(bytes);
        Ok(hex_lower(&hasher.finalize()))
    }
}

pub fn training_table_sha256() -> String {
    let data = load_training_data();
    let mut hasher = Sha256::new();
    hasher.update(TRAINING_DIGEST_DOMAIN);
    hasher.update((data.len() as u64).to_le_bytes());
    for entry in data {
        update_text(&mut hasher, entry.formula);
        hasher.update((entry.composition.len() as u64).to_le_bytes());
        for (atomic_number, fraction) in entry.composition {
            hasher.update([atomic_number]);
            hasher.update(fraction.to_bits().to_le_bytes());
        }
        hasher.update(entry.experimental_gap.to_bits().to_le_bytes());
        hasher.update([entry.crystal_system.ordinal()]);
    }
    hex_lower(&hasher.finalize())
}

pub fn predict_bandgap_evidence(
    candidate_id: CandidateId,
    composition: Vec<(u8, f64)>,
) -> Result<BandgapEvidenceReceipt, BandgapEvidenceError> {
    CandidateId::new(candidate_id.0.clone())?;
    validate_composition(&composition)?;

    let predictor = CompositionBandgapPredictor::new();
    let result = predictor.predict(&composition);
    for (name, value) in [
        ("bandgap", result.bandgap),
        ("baseline", result.baseline),
        ("ml_correction", result.ml_correction),
        ("inter_tree_stddev", result.uncertainty),
    ] {
        if !value.is_finite() {
            return Err(BandgapEvidenceError::InvalidPrediction(format!(
                "{name} must be finite"
            )));
        }
    }
    if result.bandgap < 0.0 || result.uncertainty < 0.0 {
        return Err(BandgapEvidenceError::InvalidPrediction(
            "band gap and inter-tree spread must be non-negative".into(),
        ));
    }

    let training_sha = training_table_sha256();
    let training_count = load_training_data().len();
    let mut model = ModelProvenance::named("Symthaea crystal-ablated composition RF bandgap")?;
    model.version = Some(format!(
        "v0;model-git-blob-sha1={MODEL_SOURCE_BLOB_SHA1};training-sha256={training_sha}"
    ));
    model.implementation_digest = Some(format!("git-blob-sha1:{MODEL_SOURCE_BLOB_SHA1}"));
    model.input_digest = Some(format!("sha256:{training_sha}"));

    let prediction = Prediction {
        metric: metric::BAND_GAP.into(),
        value: result.bandgap,
        unit: unit::EV.into(),
        uncertainty: UncertaintyEstimate::new(1.0, 0.0)?,
        fidelity: FidelityLevel::Surrogate,
        model,
        assumptions: vec![
            "Prediction uses composition only. Crystal-system feature 16 is fixed to CrystalSystem::Unknown for every training and inference example, so it carries no discriminative crystal information.".into(),
            "The model trains on Symthaea's curated experimental semiconductor band-gap table; this is learned surrogate evidence, not external experimental validation of the candidate.".into(),
            "The RF inter-tree standard deviation is retained in the receipt as a diagnostic only. No calibration theorem establishes it as predictive uncertainty, so generic epistemic uncertainty remains fully unknown and no interval is attached.".into(),
            "Band gap alone does not establish thermodynamic stability, absorptivity, carrier transport, defect tolerance, synthesis feasibility, toxicity, abundance, cost, device efficiency or deployability.".into(),
        ],
        evidence: vec![
            EvidenceRef {
                id: "symthaea-bandgap-curated-training-table".into(),
                kind: EvidenceKind::Dataset,
                uri: None,
                digest: Some(format!("sha256:{training_sha}")),
                note: Some(format!(
                    "training entries={training_count}; training source git blob sha1={TRAINING_SOURCE_BLOB_SHA1}"
                )),
            },
            EvidenceRef {
                id: "symthaea-bandgap-composition-rf-v0".into(),
                kind: EvidenceKind::SurrogateModel,
                uri: None,
                digest: Some(format!("git-blob-sha1:{MODEL_SOURCE_BLOB_SHA1}")),
                note: Some("crystal-ablated composition-only residual random forest".into()),
            },
        ],
    };
    prediction.validate()?;

    Ok(BandgapEvidenceReceipt {
        schema: "symthaea.bandgap.discovery-evidence.v0".into(),
        capability_classification: CAPABILITY_CLASSIFICATION.into(),
        candidate_id,
        composition,
        training_table_sha256: training_sha,
        training_entry_count: training_count,
        model_source_blob_sha1: MODEL_SOURCE_BLOB_SHA1.into(),
        training_source_blob_sha1: TRAINING_SOURCE_BLOB_SHA1.into(),
        baseline_gap_ev: result.baseline,
        ml_correction_ev: result.ml_correction,
        inter_tree_stddev_ev: result.uncertainty,
        prediction,
    })
}

fn validate_composition(composition: &[(u8, f64)]) -> Result<(), BandgapEvidenceError> {
    if composition.is_empty() {
        return Err(BandgapEvidenceError::InvalidComposition(
            "composition cannot be empty".into(),
        ));
    }
    let mut seen = BTreeSet::new();
    let mut total = 0.0;
    for &(atomic_number, fraction) in composition {
        if !(1..=103).contains(&atomic_number) {
            return Err(BandgapEvidenceError::InvalidComposition(format!(
                "atomic number {atomic_number} is outside the predictor's 1..=103 element table"
            )));
        }
        let _ = element(atomic_number);
        if !seen.insert(atomic_number) {
            return Err(BandgapEvidenceError::InvalidComposition(format!(
                "duplicate atomic number {atomic_number}"
            )));
        }
        if !fraction.is_finite() || fraction <= 0.0 {
            return Err(BandgapEvidenceError::InvalidComposition(format!(
                "atomic fraction for Z={atomic_number} must be finite and positive"
            )));
        }
        total += fraction;
    }
    if !total.is_finite() || (total - 1.0).abs() > 1e-9 {
        return Err(BandgapEvidenceError::InvalidComposition(format!(
            "composition fractions must sum to 1; got {total}"
        )));
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum BandgapEvidenceError {
    #[error("invalid composition: {0}")]
    InvalidComposition(String),
    #[error("invalid band-gap prediction: {0}")]
    InvalidPrediction(String),
    #[error(transparent)]
    Discovery(#[from] symthaea_discovery::DiscoveryError),
    #[error("band-gap evidence serialization failed: {0}")]
    Json(#[from] serde_json::Error),
}

fn update_text(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
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

    #[test]
    fn evidence_is_deterministic_and_surrogate_only() {
        let composition = vec![(14, 1.0)];
        let a = predict_bandgap_evidence(CandidateId::new("candidate-si").unwrap(), composition.clone()).unwrap();
        let b = predict_bandgap_evidence(CandidateId::new("candidate-si").unwrap(), composition).unwrap();
        assert_eq!(a.sha256().unwrap(), b.sha256().unwrap());
        assert_eq!(a.prediction.fidelity, FidelityLevel::Surrogate);
        assert!(a.prediction.evidence.iter().any(|e| e.kind == EvidenceKind::SurrogateModel));
        assert_eq!(a.prediction.uncertainty.epistemic, 1.0);
        assert!(a.prediction.uncertainty.interval.is_none());
    }

    #[test]
    fn training_identity_is_stable_and_non_empty() {
        let first = training_table_sha256();
        let second = training_table_sha256();
        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
        assert!(!load_training_data().is_empty());
    }

    #[test]
    fn malformed_compositions_fail_closed() {
        assert!(predict_bandgap_evidence(
            CandidateId::new("bad").unwrap(),
            vec![(14, 0.4), (8, 0.4)],
        )
        .is_err());
        assert!(predict_bandgap_evidence(
            CandidateId::new("dup").unwrap(),
            vec![(14, 0.5), (14, 0.5)],
        )
        .is_err());
    }
}
