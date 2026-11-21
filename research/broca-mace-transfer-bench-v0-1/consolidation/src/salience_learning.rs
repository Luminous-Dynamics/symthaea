// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Learned future-use prediction with cluster/task/source-held-out evaluation.
//! Reuse labels remain evaluator-owned; inference accepts observable features.

use std::collections::BTreeSet;
use broca_mace_policy::{Scope, SourceRootDigest};
use sha2::{Digest, Sha256};

use crate::salience::FutureUsePredictionV1;

pub const MIN_TRAIN_CLUSTERS: usize = 32;
pub const MIN_CALIBRATION_CLUSTERS: usize = 16;
pub const MIN_HELDOUT_CLUSTERS: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FeaturesV1 {
    pub recency_bps: u16,
    pub frequency_bps: u16,
    pub task_similarity_bps: u16,
    pub exact_scope_match: bool,
    pub source_support_bps: u16,
    pub measurement_quality_bps: u16,
    pub failure_memory: bool,
}
impl FeaturesV1 {
    fn valid(self) -> bool {
        [
            self.recency_bps,
            self.frequency_bps,
            self.task_similarity_bps,
            self.source_support_bps,
            self.measurement_quality_bps,
        ]
        .iter()
        .all(|value| *value <= 10_000)
    }

    fn vector(self) -> [f64; 8] {
        [
            1.0,
            f64::from(self.recency_bps) / 10_000.0,
            f64::from(self.frequency_bps) / 10_000.0,
            f64::from(self.task_similarity_bps) / 10_000.0,
            if self.exact_scope_match { 1.0 } else { 0.0 },
            f64::from(self.source_support_bps) / 10_000.0,
            f64::from(self.measurement_quality_bps) / 10_000.0,
            if self.failure_memory { 1.0 } else { 0.0 },
        ]
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExampleV1 {
    pub event_id: u32,
    pub cluster_id: u64,
    pub task_family_id: u32,
    pub source_root: SourceRootDigest,
    pub target_model_sha256: String,
    pub features: FeaturesV1,
    pub future_reused: bool,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelConfigV1 {
    pub epochs: u32,
    pub learning_rate_micros: u32,
    pub l2_micros: u32,
}
impl Default for ModelConfigV1 {
    fn default() -> Self {
        Self { epochs: 2_000, learning_rate_micros: 30_000, l2_micros: 1_000 }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Reject {
    Empty, BadId, UnknownRoot, BadFeature, DuplicateEvent, MissingClass,
    TooFewClusters, TooFewFamilies, TooFewRoots,
    ClusterOverlap, FamilyOverlap, RootOverlap, EventOverlap,
    BadConfig, ModelMismatch, MixedTargetModel, InvalidTargetModelDigest,
    TargetModelMismatch, InvalidPredictionContext, InvalidPredictionDigest,
}
#[derive(Clone, Debug, Eq, PartialEq)]
struct Identity {
    clusters: BTreeSet<u64>,
    families: BTreeSet<u32>,
    roots: BTreeSet<SourceRootDigest>,
    events: BTreeSet<u32>,
    target_model_sha256: String,
}
#[derive(Clone, Debug, Eq, PartialEq)]
struct Split {
    identity: Identity,
    digest: String,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitManifestV1 {
    pub target_model_sha256: String,
    pub train_sha256: String,
    pub calibration_sha256: String,
    pub heldout_sha256: String,
    pub task_family_split_sha256: String,
    pub source_root_split_sha256: String,
    pub training_clusters: usize,
    pub calibration_clusters: usize,
    pub heldout_clusters: usize,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ModelV1 {
    weights: [f64; 8],
    training: Identity,
    target_model_sha256: String,
    pub training_base_rate_bps: u16,
    pub model_sha256: String,
}
#[derive(Clone, Debug, PartialEq)]
pub struct CalibratorV1 {
    slope: f64,
    intercept: f64,
    model_sha256: String,
    target_model_sha256: String,
    calibration: Identity,
    pub calibration_sha256: String,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetricsV1 {
    pub examples: u32,
    pub clusters: u32,
    pub task_families: u32,
    pub source_roots: u32,
    pub brier_micros: u32,
    pub base_rate_brier_micros: u32,
    pub ece_micros: u32,
    pub worst_family_ece_micros: u32,
}
fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| {
            byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
        })
}
fn field(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}
fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value.clamp(-30.0, 30.0)).exp())
}
fn dot(weights: &[f64; 8], features: &[f64; 8]) -> f64 {
    weights.iter().zip(features).map(|(a, b)| a * b).sum()
}
fn identity(rows: &[ExampleV1]) -> Identity {
    Identity {
        clusters: rows.iter().map(|row| row.cluster_id).collect(),
        families: rows.iter().map(|row| row.task_family_id).collect(),
        roots: rows.iter().map(|row| row.source_root).collect(),
        events: rows.iter().map(|row| row.event_id).collect(),
        target_model_sha256: rows[0].target_model_sha256.clone(),
    }
}
fn dataset_digest(rows: &[ExampleV1]) -> String {
    let mut ordered: Vec<_> = rows.iter().collect();
    ordered.sort_by_key(|row| row.event_id);
    let mut hasher = Sha256::new();
    hasher.update(b"mace:future-use-dataset:v1\0");
    for row in ordered {
        hasher.update(row.event_id.to_be_bytes());
        hasher.update(row.cluster_id.to_be_bytes());
        hasher.update(row.task_family_id.to_be_bytes());
        hasher.update(row.source_root.as_bytes());
        field(&mut hasher, row.target_model_sha256.as_bytes());
        hasher.update(row.features.recency_bps.to_be_bytes());
        hasher.update(row.features.frequency_bps.to_be_bytes());
        hasher.update(row.features.task_similarity_bps.to_be_bytes());
        hasher.update([u8::from(row.features.exact_scope_match)]);
        hasher.update(row.features.source_support_bps.to_be_bytes());
        hasher.update(row.features.measurement_quality_bps.to_be_bytes());
        hasher.update([u8::from(row.features.failure_memory)]);
        hasher.update([u8::from(row.future_reused)]);
    }
    sha(&hasher.finalize())
}
fn validate(rows: &[ExampleV1], minimum_clusters: usize) -> Result<Split, Reject> {
    if rows.is_empty() { return Err(Reject::Empty); }
    let target_model = rows[0].target_model_sha256.as_str();
    if !is_lower_sha256(target_model) {
        return Err(Reject::InvalidTargetModelDigest);
    }
    let mut events = BTreeSet::new();
    let mut positive = false;
    let mut negative = false;
    for row in rows {
        if row.event_id == 0 || row.cluster_id == 0 || row.task_family_id == 0 {
            return Err(Reject::BadId);
        }
        if !row.source_root.is_known() { return Err(Reject::UnknownRoot); }
        if row.target_model_sha256 != target_model {
            return Err(Reject::MixedTargetModel);
        }
        if !row.features.valid() { return Err(Reject::BadFeature); }
        if !events.insert(row.event_id) { return Err(Reject::DuplicateEvent); }
        positive |= row.future_reused;
        negative |= !row.future_reused;
    }
    if !positive || !negative { return Err(Reject::MissingClass); }
    let ids = identity(rows);
    if ids.clusters.len() < minimum_clusters { return Err(Reject::TooFewClusters); }
    if ids.families.len() < 4 { return Err(Reject::TooFewFamilies); }
    if ids.roots.len() < minimum_clusters / 2 { return Err(Reject::TooFewRoots); }
    Ok(Split { identity: ids, digest: dataset_digest(rows) })
}
fn ensure_disjoint(left: &Identity, right: &Identity) -> Result<(), Reject> {
    if !left.clusters.is_disjoint(&right.clusters) { return Err(Reject::ClusterOverlap); }
    if !left.families.is_disjoint(&right.families) { return Err(Reject::FamilyOverlap); }
    if !left.roots.is_disjoint(&right.roots) { return Err(Reject::RootOverlap); }
    if !left.events.is_disjoint(&right.events) { return Err(Reject::EventOverlap); }
    Ok(())
}
pub fn qualify_splits(
    training: &[ExampleV1],
    calibration: &[ExampleV1],
    heldout: &[ExampleV1],
) -> Result<SplitManifestV1, Reject> {
    let train = validate(training, MIN_TRAIN_CLUSTERS)?;
    let cal = validate(calibration, MIN_CALIBRATION_CLUSTERS)?;
    let test = validate(heldout, MIN_HELDOUT_CLUSTERS)?;
    if train.identity.target_model_sha256 != cal.identity.target_model_sha256
        || train.identity.target_model_sha256 != test.identity.target_model_sha256
    {
        return Err(Reject::MixedTargetModel);
    }
    ensure_disjoint(&train.identity, &cal.identity)?;
    ensure_disjoint(&train.identity, &test.identity)?;
    ensure_disjoint(&cal.identity, &test.identity)?;
    let mut family_hasher = Sha256::new();
    family_hasher.update(b"mace:task-family-split:v1\\0");
    let mut root_hasher = Sha256::new();
    root_hasher.update(b"mace:source-root-split:v1\\0");
    for (label, identity) in [
        ("train", &train.identity),
        ("calibration", &cal.identity),
        ("heldout", &test.identity),
    ] {
        field(&mut family_hasher, label.as_bytes());
        for family in &identity.families {
            family_hasher.update(family.to_be_bytes());
        }
        field(&mut root_hasher, label.as_bytes());
        for root in &identity.roots {
            root_hasher.update(root.as_bytes());
        }
    }
    Ok(SplitManifestV1 {
        target_model_sha256: train.identity.target_model_sha256.clone(),
        train_sha256: train.digest,
        calibration_sha256: cal.digest,
        heldout_sha256: test.digest,
        task_family_split_sha256: sha(&family_hasher.finalize()),
        source_root_split_sha256: sha(&root_hasher.finalize()),
        training_clusters: train.identity.clusters.len(),
        calibration_clusters: cal.identity.clusters.len(),
        heldout_clusters: test.identity.clusters.len(),
    })
}
pub fn train(rows: &[ExampleV1], config: ModelConfigV1) -> Result<ModelV1, Reject> {
    if config.epochs == 0 || config.epochs > 50_000
        || !(1_000..=200_000).contains(&config.learning_rate_micros)
        || config.l2_micros > 100_000 {
        return Err(Reject::BadConfig);
    }
    let split = validate(rows, MIN_TRAIN_CLUSTERS)?;
    let mut ordered: Vec<_> = rows.iter().collect();
    ordered.sort_by_key(|row| row.event_id);
    let count = ordered.len() as f64;
    let positives = ordered.iter().filter(|row| row.future_reused).count() as f64;
    let training_base_rate_bps = (10_000.0 * positives / count)
        .round()
        .clamp(0.0, 10_000.0) as u16;
    let rate = f64::from(config.learning_rate_micros) / 1_000_000.0;
    let l2 = f64::from(config.l2_micros) / 1_000_000.0;
    let mut weights = [0.0_f64; 8];
    for _ in 0..config.epochs {
        let mut gradient = [0.0_f64; 8];
        for row in &ordered {
            let features = row.features.vector();
            let target = if row.future_reused { 1.0 } else { 0.0 };
            let error = sigmoid(dot(&weights, &features)) - target;
            for index in 0..8 { gradient[index] += error * features[index]; }
        }
        for index in 0..8 {
            let penalty = if index == 0 { 0.0 } else { l2 * weights[index] };
            weights[index] = (
                weights[index] - rate * (gradient[index] / count + penalty)
            ).clamp(-12.0, 12.0);
        }
    }
    let mut hasher = Sha256::new();
    hasher.update(b"mace:future-use-model:v1\0");
    field(&mut hasher, split.digest.as_bytes());
    field(&mut hasher, split.identity.target_model_sha256.as_bytes());
    hasher.update(config.epochs.to_be_bytes());
    hasher.update(config.learning_rate_micros.to_be_bytes());
    hasher.update(config.l2_micros.to_be_bytes());
    hasher.update(training_base_rate_bps.to_be_bytes());
    for weight in weights { hasher.update(weight.to_bits().to_be_bytes()); }
    Ok(ModelV1 {
        weights,
        target_model_sha256: split.identity.target_model_sha256.clone(),
        training: split.identity,
        training_base_rate_bps,
        model_sha256: sha(&hasher.finalize()),
    })
}
impl ModelV1 {
    pub fn target_model_sha256(&self) -> &str {
        &self.target_model_sha256
    }

    fn logit(&self, features: FeaturesV1) -> Result<f64, Reject> {
        if !features.valid() { return Err(Reject::BadFeature); }
        Ok(dot(&self.weights, &features.vector()))
    }
}
pub fn calibrate(model: &ModelV1, rows: &[ExampleV1]) -> Result<CalibratorV1, Reject> {
    let split = validate(rows, MIN_CALIBRATION_CLUSTERS)?;
    if model.target_model_sha256 != split.identity.target_model_sha256 {
        return Err(Reject::MixedTargetModel);
    }
    ensure_disjoint(&model.training, &split.identity)?;
    let mut ordered: Vec<_> = rows.iter().collect();
    ordered.sort_by_key(|row| row.event_id);
    let count = ordered.len() as f64;
    let mut slope = 1.0_f64;
    let mut intercept = 0.0_f64;
    for _ in 0..2_000 {
        let mut slope_gradient = 0.0;
        let mut intercept_gradient = 0.0;
        for row in &ordered {
            let logit = model.logit(row.features)?;
            let target = if row.future_reused { 1.0 } else { 0.0 };
            let error = sigmoid(slope * logit + intercept) - target;
            slope_gradient += error * logit;
            intercept_gradient += error;
        }
        slope = (slope - 0.02 * slope_gradient / count).clamp(0.05, 20.0);
        intercept = (intercept - 0.02 * intercept_gradient / count).clamp(-10.0, 10.0);
    }
    let mut hasher = Sha256::new();
    hasher.update(b"mace:future-use-calibrator:v1\0");
    field(&mut hasher, model.model_sha256.as_bytes());
    field(&mut hasher, model.target_model_sha256.as_bytes());
    field(&mut hasher, split.digest.as_bytes());
    hasher.update(slope.to_bits().to_be_bytes());
    hasher.update(intercept.to_bits().to_be_bytes());
    Ok(CalibratorV1 {
        slope,
        intercept,
        model_sha256: model.model_sha256.clone(),
        target_model_sha256: model.target_model_sha256.clone(),
        calibration: split.identity,
        calibration_sha256: sha(&hasher.finalize()),
    })
}
impl CalibratorV1 {
    pub fn target_model_sha256(&self) -> &str {
        &self.target_model_sha256
    }

    pub fn predict_bps(&self, model: &ModelV1, features: FeaturesV1) -> Result<u16, Reject> {
        if self.model_sha256 != model.model_sha256 {
            return Err(Reject::ModelMismatch);
        }
        if self.target_model_sha256 != model.target_model_sha256 {
            return Err(Reject::TargetModelMismatch);
        }
        let probability = sigmoid(self.slope * model.logit(features)? + self.intercept);
        Ok((probability * 10_000.0).round().clamp(0.0, 10_000.0) as u16)
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetricsV1 {
    pub examples: u32,
    pub clusters: u32,
    pub task_families: u32,
    pub source_roots: u32,
    pub brier_micros: u32,
    pub base_rate_brier_micros: u32,
    pub ece_micros: u32,
    pub worst_family_ece_micros: u32,
}
fn ece(rows: &[ExampleV1], probabilities: &[f64]) -> f64 {
    let mut counts = [0_u32; 10];
    let mut predicted = [0.0_f64; 10];
    let mut observed = [0.0_f64; 10];
    for (row, probability) in rows.iter().zip(probabilities) {
        let bin = ((*probability * 10.0).floor() as usize).min(9);
        counts[bin] += 1;
        predicted[bin] += *probability;
        observed[bin] += if row.future_reused { 1.0 } else { 0.0 };
    }
    let total = rows.len() as f64;
    let mut result = 0.0;
    for index in 0..10 {
        if counts[index] > 0 {
            let count = f64::from(counts[index]);
            result += count / total
                * (predicted[index] / count - observed[index] / count).abs();
        }
    }
    result
}
pub fn evaluate(
    model: &ModelV1,
    calibrator: &CalibratorV1,
    rows: &[ExampleV1],
) -> Result<MetricsV1, Reject> {
    if calibrator.model_sha256 != model.model_sha256 {
        return Err(Reject::ModelMismatch);
    }
    if calibrator.target_model_sha256 != model.target_model_sha256 {
        return Err(Reject::TargetModelMismatch);
    }
    let split = validate(rows, MIN_HELDOUT_CLUSTERS)?;
    if split.identity.target_model_sha256 != model.target_model_sha256 {
        return Err(Reject::MixedTargetModel);
    }
    ensure_disjoint(&model.training, &split.identity)?;
    ensure_disjoint(&calibrator.calibration, &split.identity)?;
    let probabilities: Vec<f64> = rows.iter().map(|row| {
        calibrator.predict_bps(model, row.features).map(|value| f64::from(value) / 10_000.0)
    }).collect::<Result<_, _>>()?;
    // Deployable reference uses only training labels; held-out labels are
    // reserved for evaluation and cannot set the comparator's prevalence.
    let training_base_rate =
        f64::from(model.training_base_rate_bps) / 10_000.0;
    let mut brier = 0.0;
    let mut base_brier = 0.0;
    for (row, probability) in rows.iter().zip(&probabilities) {
        let target = if row.future_reused { 1.0 } else { 0.0 };
        brier += (*probability - target).powi(2);
        base_brier += (training_base_rate - target).powi(2);
    }
    brier /= rows.len() as f64;
    base_brier /= rows.len() as f64;
    let overall_ece = ece(rows, &probabilities);
    let mut groups: std::collections::BTreeMap<u32, Vec<usize>> = Default::default();
    for (index, row) in rows.iter().enumerate() {
        groups.entry(row.task_family_id).or_default().push(index);
    }
    let mut worst = 0.0_f64;
    for indices in groups.values() {
        if indices.len() < 4 { continue; }
        let group_rows: Vec<_> = indices.iter().map(|i| rows[*i].clone()).collect();
        let group_probs: Vec<_> = indices.iter().map(|i| probabilities[*i]).collect();
        worst = worst.max(ece(&group_rows, &group_probs));
    }
    let micros = |value: f64| {
        (value * 1_000_000.0).round().clamp(0.0, 1_000_000.0) as u32
    };
    Ok(MetricsV1 {
        examples: rows.len() as u32,
        clusters: split.identity.clusters.len() as u32,
        task_families: split.identity.families.len() as u32,
        source_roots: split.identity.roots.len() as u32,
        brier_micros: micros(brier),
        base_rate_brier_micros: micros(base_brier),
        ece_micros: micros(overall_ece),
        worst_family_ece_micros: micros(worst),
    })
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawFutureUsePredictionInputV1 {
    pub event_id: u32,
    pub target_model_sha256: String,
    pub target_scope: Scope,
    pub predicted_at_ms: i64,
    pub horizon_end_ms: i64,
    pub expected_value_gain_micros: u32,
    pub uncertainty_penalty_micros: u32,
    pub rationale_sha256: String,
    pub evidence_sha256: String,
}

/// Build a raw candidate prediction from the learned model and its calibrator.
/// The returned prediction is NOT eligible for surfacing until the separate
/// SaliencePredictionVerifierV1 issues a valid calibration receipt.
pub fn build_raw_prediction(
    model: &ModelV1,
    calibrator: &CalibratorV1,
    features: FeaturesV1,
    input: RawFutureUsePredictionInputV1,
) -> Result<FutureUsePredictionV1, Reject> {
    if input.event_id == 0
        || !is_lower_sha256(&input.target_model_sha256)
        || input.predicted_at_ms < 0
        || input.horizon_end_ms <= input.predicted_at_ms
        || input.rationale_sha256.len() != 64
        || input.evidence_sha256.len() != 64
        || !input.rationale_sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        || !input.evidence_sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(Reject::InvalidPredictionContext);
    }
    if input.target_model_sha256 != model.target_model_sha256
        || calibrator.target_model_sha256 != input.target_model_sha256
    {
        return Err(Reject::TargetModelMismatch);
    }
    let probability = calibrator.predict_bps(model, features)?;
    let mut h = Sha256::new();
    h.update(b"mace:learned-salience-pipeline:v1\0");
    field(&mut h, model.model_sha256.as_bytes());
    field(&mut h, model.target_model_sha256.as_bytes());
    field(&mut h, calibrator.calibration_sha256.as_bytes());
    let pipeline_digest = sha(&h.finalize());

    Ok(FutureUsePredictionV1::new(
        input.event_id,
        input.target_model_sha256,
        input.target_scope,
        input.predicted_at_ms,
        input.horizon_end_ms,
        probability,
        input.expected_value_gain_micros,
        input.uncertainty_penalty_micros,
        "mace-learned-future-use",
        "logistic-platt-v1",
        pipeline_digest,
        input.rationale_sha256,
        input.evidence_sha256,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rows(event: u32, cluster: u64, family: u32) -> Vec<ExampleV1> {
        let mut result = Vec::new();
        for c in 0..32_u32 {
            for j in 0..8_u32 {
                let id = event + c * 8 + j + 1;
                let mut root = [0_u8; 32];
                root[..4].copy_from_slice(&(id + 50_000).to_be_bytes());
                result.push(ExampleV1 {
                    event_id: id,
                    cluster_id: cluster + u64::from(c) + 1,
                    task_family_id: family + c % 8,
                    source_root: SourceRootDigest::from_bytes(root),
                    target_model_sha256: sha(b"unit-test-target-model-v1"),
                    features: FeaturesV1 {
                        recency_bps: 7_000,
                        frequency_bps: 6_000,
                        task_similarity_bps: 8_000,
                        exact_scope_match: true,
                        source_support_bps: 7_000,
                        measurement_quality_bps: 9_000,
                        failure_memory: false,
                    },
                    future_reused: (c + j) % 3 != 0,
                });
            }
        }
        result
    }
    #[test]
    fn train_calibration_and_heldout_splits_are_disjoint() {
        let train = rows(1, 1, 10);
        let calibration = rows(10_000, 100, 30);
        let heldout = rows(20_000, 200, 50);
        let manifest = qualify_splits(&train, &calibration, &heldout).unwrap();
        assert_eq!(manifest.training_clusters, 32);
        assert_eq!(manifest.heldout_clusters, 32);
    }
    #[test]
    fn source_root_overlap_is_rejected_across_splits() {
        let train = rows(1, 1, 10);
        let calibration = rows(10_000, 100, 30);
        let mut heldout = rows(20_000, 200, 50);
        heldout[0].source_root = train[0].source_root;
        assert_eq!(
            qualify_splits(&train, &calibration, &heldout),
            Err(Reject::RootOverlap)
        );
    }
    #[test]
    fn learned_model_calibrates_and_reports_heldout_metrics() {
        let training_rows = rows(1, 1, 10);
        let calibration = rows(10_000, 100, 30);
        let heldout = rows(20_000, 200, 50);
        let model = train(&training_rows, ModelConfigV1::default()).unwrap();
        let calibrator = calibrate(&model, &calibration).unwrap();
        let metrics = evaluate(&model, &calibrator, &heldout).unwrap();
        assert_eq!(metrics.examples, heldout.len() as u32);
        assert_eq!(metrics.clusters, 32);
    }
    #[test]
    fn learned_output_is_raw_prediction_not_self_verified_memory() {
        let training = rows(1, 1, 10);
        let calibration = rows(10_000, 100, 30);
        let model = train(&training, ModelConfigV1::default()).unwrap();
        let calibrator = calibrate(&model, &calibration).unwrap();
        let prediction = build_raw_prediction(
            &model,
            &calibrator,
            training[0].features,
            RawFutureUsePredictionInputV1 {
                event_id: 77,
                target_scope: Scope {
                    regime: broca_mace_policy::Regime::Baseline,
                    objective: broca_mace_policy::Objective::Quality,
                },
                predicted_at_ms: 100,
                horizon_end_ms: 200,
                expected_value_gain_micros: 500,
                uncertainty_penalty_micros: 50,
                target_model_sha256: model.target_model_sha256().to_owned(),
                rationale_sha256: "a".repeat(64),
                evidence_sha256: "b".repeat(64),
            },
        )
        .unwrap();
        assert_eq!(prediction.prediction_sha256, prediction.canonical_digest());
        assert_eq!(prediction.model_id, "mace-learned-future-use");
        assert_eq!(prediction.model_version, "logistic-platt-v1");
    }

    #[test]
    fn split_manifest_binds_task_family_and_source_root_membership() {
        let training = rows(1, 1, 10);
        let calibration = rows(10_000, 100, 30);
        let heldout = rows(20_000, 200, 50);
        let manifest = qualify_splits(&training, &calibration, &heldout).unwrap();
        assert_eq!(manifest.task_family_split_sha256.len(), 64);
        assert_eq!(manifest.source_root_split_sha256.len(), 64);
    }

    #[test]
    fn heldout_prevalence_does_not_change_training_base_rate_comparator() {
        let training = rows(1, 1, 10);
        let calibration = rows(10_000, 100, 30);
        let mut heldout_a = rows(20_000, 200, 50);
        let mut heldout_b = heldout_a.clone();
        for row in &mut heldout_b {
            row.future_reused = !row.future_reused;
        }
        let model = train(&training, ModelConfigV1::default()).unwrap();
        let calibrator = calibrate(&model, &calibration).unwrap();
        let score_a = evaluate(&model, &calibrator, &heldout_a).unwrap();
        let score_b = evaluate(&model, &calibrator, &heldout_b).unwrap();
        assert_eq!(
            score_a.base_rate_brier_micros,
            score_b.base_rate_brier_micros
        );
    }

    #[test]
    fn model_identity_must_match_across_train_calibration_and_heldout() {
        let training = rows(1, 1, 10);
        let mut calibration = rows(10_000, 100, 30);
        let heldout = rows(20_000, 200, 50);
        calibration[0].target_model_sha256 = sha(b"other-target-model");
        assert_eq!(
            qualify_splits(&training, &calibration, &heldout),
            Err(Reject::MixedTargetModel)
        );
    }

    #[test]
    fn a_training_batch_cannot_mix_target_models() {
        let mut training = rows(1, 1, 10);
        training[0].target_model_sha256 = sha(b"other-target-model");
        assert_eq!(
            train(&training, ModelConfigV1::default()),
            Err(Reject::MixedTargetModel)
        );
    }

    #[test]
    fn raw_prediction_is_bound_to_the_model_it_will_be_sent_to() {
        let training = rows(1, 1, 10);
        let calibration = rows(10_000, 100, 30);
        let model = train(&training, ModelConfigV1::default()).unwrap();
        let calibrator = calibrate(&model, &calibration).unwrap();
        let result = build_raw_prediction(
            &model,
            &calibrator,
            training[0].features,
            RawFutureUsePredictionInputV1 {
                event_id: 78,
                target_model_sha256: sha(b"another-target-model"),
                target_scope: Scope {
                    regime: broca_mace_policy::Regime::Baseline,
                    objective: broca_mace_policy::Objective::Quality,
                },
                predicted_at_ms: 100,
                horizon_end_ms: 200,
                expected_value_gain_micros: 500,
                uncertainty_penalty_micros: 50,
                rationale_sha256: "c".repeat(64),
                evidence_sha256: "d".repeat(64),
            },
        );
        assert_eq!(result, Err(Reject::TargetModelMismatch));
    }

    #[test]
    fn calibrator_is_bound_to_the_exact_model() {
        let train_rows = rows(1, 1, 10);
        let calibration = rows(10_000, 100, 30);
        let model = train(&train_rows, ModelConfigV1::default()).unwrap();
        let calibrator = calibrate(&model, &calibration).unwrap();
        let changed = train(
            &train_rows,
            ModelConfigV1 { epochs: 100, ..ModelConfigV1::default() },
        ).unwrap();
        assert_eq!(
            calibrator.predict_bps(&changed, train_rows[0].features),
            Err(Reject::ModelMismatch)
        );
    }
}
