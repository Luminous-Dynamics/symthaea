// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Leakage-aware, measurement-only protocol for Energy Discovery Benchmark Zero.
//!
//! This crate evaluates an already-ranked band-gap screening run against an
//! exact truth-data slice. It deliberately performs no model training, material
//! generation, solver invocation, experiment, or deployment action.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const PROTOCOL_VERSION: &str = "energy-benchmark-zero.v0";
pub const CAPABILITY_CLASSIFICATION: &str =
    "MEASUREMENT-ONLY BENCHMARK RECEIPT -- not a material certification, novelty claim, experiment authorization, or deployment decision.";

const RANKING_DIGEST_DOMAIN: &[u8] = b"symthaea.energy-benchmark-zero.ranking.v0\0";
const RECEIPT_DIGEST_DOMAIN: &[u8] = b"symthaea.energy-benchmark-zero.receipt.v0\0";

/// Exact logical slice of a dataset. Train/test folds from the same parent
/// dataset are distinct slices; reusing the exact truth slice for training is
/// forbidden by this protocol.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DataSliceId {
    pub dataset_id: String,
    pub split_id: String,
}

impl DataSliceId {
    pub fn new(
        dataset_id: impl Into<String>,
        split_id: impl Into<String>,
    ) -> Result<Self, BenchmarkError> {
        let value = Self {
            dataset_id: dataset_id.into(),
            split_id: split_id.into(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), BenchmarkError> {
        if self.dataset_id.trim().is_empty() || self.split_id.trim().is_empty() {
            return Err(BenchmarkError::Invalid(
                "dataset and split identifiers cannot be empty".into(),
            ));
        }
        Ok(())
    }
}

/// Provenance required for the exact truth slice used by a benchmark.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetProvenance {
    pub slice: DataSliceId,
    pub source_uri: String,
    pub content_digest: String,
    pub license: String,
}

impl DatasetProvenance {
    pub fn validate(&self) -> Result<(), BenchmarkError> {
        self.slice.validate()?;
        for (name, value) in [
            ("source URI", self.source_uri.as_str()),
            ("content digest", self.content_digest.as_str()),
            ("license", self.license.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(BenchmarkError::Invalid(format!(
                    "truth dataset {name} cannot be empty"
                )));
            }
        }
        Ok(())
    }
}

/// Declared provenance for the ranking method. This is a disclosure contract,
/// not a proof that the caller listed every source of prior knowledge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScreeningMethodProvenance {
    pub method_id: String,
    pub version: String,
    #[serde(default)]
    pub training_slices: BTreeSet<DataSliceId>,
}

impl ScreeningMethodProvenance {
    pub fn validate(&self) -> Result<(), BenchmarkError> {
        if self.method_id.trim().is_empty() || self.version.trim().is_empty() {
            return Err(BenchmarkError::Invalid(
                "screening method id and version cannot be empty".into(),
            ));
        }
        for slice in &self.training_slices {
            slice.validate()?;
        }
        Ok(())
    }
}

/// Caller-supplied target window. The protocol intentionally embeds no claim
/// about a universally optimal photovoltaic band gap.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BandgapTarget {
    pub min_ev: f64,
    pub max_ev: f64,
}

impl BandgapTarget {
    pub fn new(min_ev: f64, max_ev: f64) -> Result<Self, BenchmarkError> {
        let value = Self { min_ev, max_ev };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), BenchmarkError> {
        if !self.min_ev.is_finite()
            || !self.max_ev.is_finite()
            || self.min_ev < 0.0
            || self.min_ev >= self.max_ev
        {
            return Err(BenchmarkError::Invalid(
                "band-gap target requires finite non-negative min < max".into(),
            ));
        }
        Ok(())
    }

    pub fn midpoint(&self) -> f64 {
        0.5 * (self.min_ev + self.max_ev)
    }

    pub fn qualifies(&self, gap_ev: f64) -> bool {
        gap_ev >= self.min_ev && gap_ev <= self.max_ev
    }

    pub fn midpoint_error(&self, gap_ev: f64) -> f64 {
        (gap_ev - self.midpoint()).abs()
    }
}

/// One item in the method-produced ranking. Experimental truth is deliberately
/// absent from this type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScreeningRecord {
    pub candidate_id: String,
    pub predicted_gap_ev: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncertainty_ev: Option<f64>,
}

impl ScreeningRecord {
    fn validate(&self) -> Result<(), BenchmarkError> {
        if self.candidate_id.trim().is_empty() {
            return Err(BenchmarkError::Invalid(
                "screening candidate id cannot be empty".into(),
            ));
        }
        if !self.predicted_gap_ev.is_finite() || self.predicted_gap_ev < 0.0 {
            return Err(BenchmarkError::Invalid(format!(
                "prediction for {:?} must be finite and non-negative",
                self.candidate_id
            )));
        }
        if let Some(uncertainty) = self.uncertainty_ev {
            if !uncertainty.is_finite() || uncertainty < 0.0 {
                return Err(BenchmarkError::Invalid(format!(
                    "uncertainty for {:?} must be finite and non-negative",
                    self.candidate_id
                )));
            }
        }
        Ok(())
    }
}

/// Ordered output of a screening method, best candidate first.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScreeningRun {
    pub method: ScreeningMethodProvenance,
    pub target: BandgapTarget,
    pub ranked: Vec<ScreeningRecord>,
}

impl ScreeningRun {
    pub fn validate(&self) -> Result<(), BenchmarkError> {
        self.method.validate()?;
        self.target.validate()?;
        if self.ranked.is_empty() {
            return Err(BenchmarkError::Invalid(
                "screening ranking cannot be empty".into(),
            ));
        }
        let mut seen = BTreeSet::new();
        for record in &self.ranked {
            record.validate()?;
            if !seen.insert(record.candidate_id.as_str()) {
                return Err(BenchmarkError::DuplicateCandidate(
                    record.candidate_id.clone(),
                ));
            }
        }
        Ok(())
    }

    /// Content identity for the exact ordered ranking, including predictions
    /// and per-item uncertainty. This is deliberately order-sensitive.
    pub fn ranking_digest(&self) -> Result<String, BenchmarkError> {
        self.validate()?;
        domain_separated_digest(RANKING_DIGEST_DOMAIN, &self.ranked)
    }
}

/// Experimental/reference truth kept outside the screening records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BandgapTruthSet {
    pub provenance: DatasetProvenance,
    pub experimental_gap_ev: BTreeMap<String, f64>,
}

impl BandgapTruthSet {
    pub fn validate(&self) -> Result<(), BenchmarkError> {
        self.provenance.validate()?;
        if self.experimental_gap_ev.is_empty() {
            return Err(BenchmarkError::Invalid(
                "truth set cannot be empty".into(),
            ));
        }
        for (candidate, gap) in &self.experimental_gap_ev {
            if candidate.trim().is_empty() || !gap.is_finite() || *gap < 0.0 {
                return Err(BenchmarkError::Invalid(
                    "truth entries require non-empty ids and finite non-negative gaps".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    pub k: usize,
    pub ranked_candidate_count: usize,
    pub truth_candidate_count: usize,
    pub qualifying_truth_count: usize,
    pub top_k_hits: usize,
    pub top_k_precision: f64,
    pub top_k_recall: f64,
    pub mean_abs_prediction_error_ev: f64,
    pub best_selected_target_error_ev: f64,
    pub global_best_target_error_ev: f64,
    pub target_regret_ev: f64,
}

/// Deterministic review artifact for one benchmark measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkReceipt {
    pub protocol_version: &'static str,
    pub capability_classification: &'static str,
    pub truth: DatasetProvenance,
    pub method: ScreeningMethodProvenance,
    pub target: BandgapTarget,
    /// Domain-separated identity of the exact ordered method output used to
    /// compute these metrics.
    pub screening_output_digest: String,
    pub metrics: BenchmarkMetrics,
}

impl BenchmarkReceipt {
    pub fn digest(&self) -> Result<String, BenchmarkError> {
        domain_separated_digest(RECEIPT_DIGEST_DOMAIN, self)
    }

    pub fn to_json_pretty(&self) -> Result<String, BenchmarkError> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

fn domain_separated_digest<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<String, BenchmarkError> {
    let bytes = serde_json::to_vec(value)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

/// Evaluate the first `k` ranked candidates without exposing truth to the
/// ranking representation itself.
pub fn evaluate(
    run: &ScreeningRun,
    truth: &BandgapTruthSet,
    k: usize,
) -> Result<BenchmarkReceipt, BenchmarkError> {
    run.validate()?;
    truth.validate()?;

    if run.method.training_slices.contains(&truth.provenance.slice) {
        return Err(BenchmarkError::TrainingTruthOverlap(
            truth.provenance.slice.clone(),
        ));
    }
    if k == 0 || k > run.ranked.len() {
        return Err(BenchmarkError::InvalidK {
            requested: k,
            available: run.ranked.len(),
        });
    }

    let qualifying_truth_count = truth
        .experimental_gap_ev
        .values()
        .filter(|&&gap| run.target.qualifies(gap))
        .count();
    if qualifying_truth_count == 0 {
        return Err(BenchmarkError::NoQualifyingTruth);
    }

    let mut absolute_error_sum = 0.0;
    for record in &run.ranked {
        let observed = *truth
            .experimental_gap_ev
            .get(&record.candidate_id)
            .ok_or_else(|| BenchmarkError::MissingTruth(record.candidate_id.clone()))?;
        absolute_error_sum += (record.predicted_gap_ev - observed).abs();
    }

    let top_k_hits = run.ranked[..k]
        .iter()
        .filter(|record| {
            let observed = truth.experimental_gap_ev[&record.candidate_id];
            run.target.qualifies(observed)
        })
        .count();

    let best_selected_target_error_ev = run.ranked[..k]
        .iter()
        .map(|record| {
            run.target
                .midpoint_error(truth.experimental_gap_ev[&record.candidate_id])
        })
        .fold(f64::INFINITY, f64::min);
    let global_best_target_error_ev = truth
        .experimental_gap_ev
        .values()
        .map(|&gap| run.target.midpoint_error(gap))
        .fold(f64::INFINITY, f64::min);

    let metrics = BenchmarkMetrics {
        k,
        ranked_candidate_count: run.ranked.len(),
        truth_candidate_count: truth.experimental_gap_ev.len(),
        qualifying_truth_count,
        top_k_hits,
        top_k_precision: top_k_hits as f64 / k as f64,
        top_k_recall: top_k_hits as f64 / qualifying_truth_count as f64,
        mean_abs_prediction_error_ev: absolute_error_sum / run.ranked.len() as f64,
        best_selected_target_error_ev,
        global_best_target_error_ev,
        target_regret_ev: (best_selected_target_error_ev - global_best_target_error_ev).max(0.0),
    };

    Ok(BenchmarkReceipt {
        protocol_version: PROTOCOL_VERSION,
        capability_classification: CAPABILITY_CLASSIFICATION,
        truth: truth.provenance.clone(),
        method: run.method.clone(),
        target: run.target,
        screening_output_digest: run.ranking_digest()?,
        metrics,
    })
}

#[derive(Debug, Error)]
pub enum BenchmarkError {
    #[error("invalid benchmark contract: {0}")]
    Invalid(String),
    #[error("screening ranking contains duplicate candidate {0:?}")]
    DuplicateCandidate(String),
    #[error("truth is missing ranked candidate {0:?}")]
    MissingTruth(String),
    #[error("requested top-k {requested} is invalid for {available} ranked candidates")]
    InvalidK { requested: usize, available: usize },
    #[error("truth set has no candidates inside the requested target window")]
    NoQualifyingTruth,
    #[error("benchmark truth slice was also declared as training data: {0:?}")]
    TrainingTruthOverlap(DataSliceId),
    #[error("benchmark receipt serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slice(split: &str) -> DataSliceId {
        DataSliceId::new("external-gap-v1", split).unwrap()
    }

    fn truth() -> BandgapTruthSet {
        BandgapTruthSet {
            provenance: DatasetProvenance {
                slice: slice("test-0"),
                source_uri: "https://example.invalid/external-gap-v1".into(),
                content_digest: "sha256:truth".into(),
                license: "test-fixture".into(),
            },
            experimental_gap_ev: BTreeMap::from([
                ("A".into(), 1.30),
                ("B".into(), 2.00),
                ("C".into(), 1.40),
            ]),
        }
    }

    fn run() -> ScreeningRun {
        ScreeningRun {
            method: ScreeningMethodProvenance {
                method_id: "toy-blind-ranker".into(),
                version: "v1".into(),
                training_slices: BTreeSet::from([slice("train-0")]),
            },
            target: BandgapTarget::new(1.20, 1.50).unwrap(),
            ranked: vec![
                ScreeningRecord {
                    candidate_id: "A".into(),
                    predicted_gap_ev: 1.35,
                    uncertainty_ev: Some(0.10),
                },
                ScreeningRecord {
                    candidate_id: "C".into(),
                    predicted_gap_ev: 1.50,
                    uncertainty_ev: None,
                },
                ScreeningRecord {
                    candidate_id: "B".into(),
                    predicted_gap_ev: 1.90,
                    uncertainty_ev: Some(0.20),
                },
            ],
        }
    }

    #[test]
    fn exact_training_truth_slice_overlap_fails_closed() {
        let mut run = run();
        run.method.training_slices.insert(slice("test-0"));
        assert!(matches!(
            evaluate(&run, &truth(), 2),
            Err(BenchmarkError::TrainingTruthOverlap(_))
        ));
    }

    #[test]
    fn clean_blind_ranking_produces_measurement_only_metrics() {
        let receipt = evaluate(&run(), &truth(), 2).unwrap();
        assert_eq!(receipt.protocol_version, PROTOCOL_VERSION);
        assert_eq!(receipt.capability_classification, CAPABILITY_CLASSIFICATION);
        assert_eq!(receipt.metrics.top_k_hits, 2);
        assert_eq!(receipt.metrics.top_k_precision, 1.0);
        assert_eq!(receipt.metrics.top_k_recall, 1.0);
        assert!(receipt.metrics.mean_abs_prediction_error_ev > 0.0);
        assert!(receipt.metrics.target_regret_ev <= 1e-12);
        assert!(!receipt.screening_output_digest.is_empty());
    }

    #[test]
    fn missing_truth_for_any_ranked_candidate_is_rejected() {
        let mut truth = truth();
        truth.experimental_gap_ev.remove("B");
        assert!(matches!(
            evaluate(&run(), &truth, 2),
            Err(BenchmarkError::MissingTruth(candidate)) if candidate == "B"
        ));
    }

    #[test]
    fn duplicate_ranked_candidate_is_rejected() {
        let mut run = run();
        run.ranked.push(run.ranked[0].clone());
        assert!(matches!(
            evaluate(&run, &truth(), 2),
            Err(BenchmarkError::DuplicateCandidate(candidate)) if candidate == "A"
        ));
    }

    #[test]
    fn invalid_target_or_prediction_fails_closed() {
        assert!(BandgapTarget::new(f64::NAN, 1.5).is_err());
        assert!(BandgapTarget::new(1.5, 1.5).is_err());

        let mut run = run();
        run.ranked[0].predicted_gap_ev = f64::INFINITY;
        assert!(evaluate(&run, &truth(), 2).is_err());
    }

    #[test]
    fn receipt_digest_is_deterministic_and_binds_exact_ranking() {
        let receipt_a = evaluate(&run(), &truth(), 2).unwrap();
        let receipt_b = evaluate(&run(), &truth(), 2).unwrap();
        assert_eq!(receipt_a.digest().unwrap(), receipt_b.digest().unwrap());
        assert_eq!(
            receipt_a.screening_output_digest,
            receipt_b.screening_output_digest
        );

        let mut changed = run();
        changed.ranked.swap(0, 1);
        let changed_receipt = evaluate(&changed, &truth(), 2).unwrap();
        // Summary metrics can remain equal when two qualifying candidates trade
        // places; the exact ordered output must still be a different artifact.
        assert_ne!(
            receipt_a.screening_output_digest,
            changed_receipt.screening_output_digest
        );
        assert_ne!(receipt_a.digest().unwrap(), changed_receipt.digest().unwrap());
    }

    #[test]
    fn top_k_must_be_explicit_and_bounded() {
        assert!(matches!(
            evaluate(&run(), &truth(), 0),
            Err(BenchmarkError::InvalidK { .. })
        ));
        assert!(matches!(
            evaluate(&run(), &truth(), 4),
            Err(BenchmarkError::InvalidK { .. })
        ));
    }
}
