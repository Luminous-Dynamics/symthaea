// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Composition-only learned band-gap predictor.
//!
//! Unlike [`crate::ml_bandgap::BandgapPredictor`], this model never consumes a
//! crystal-system label. Training and inference both canonicalize composition
//! through one 1e-9 atomic-fraction identity and mask crystal information to
//! [`CrystalSystem::Unknown`].

#![forbid(unsafe_code)]

use crate::bandgap_baseline::electronegativity_bandgap;
use crate::features::{N_FEATURES, extract_features};
use crate::ml_bandgap::BandgapPrediction;
use crate::training_data::{CrystalSystem, MaterialEntry, load_training_data};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

pub const MODEL_ID: &str = "symthaea.bandgap.composition-only-rf-residual";
pub const MODEL_VERSION: &str = "v0";
pub const FEATURE_SCHEMA: &str = "symthaea.bandgap.features18.crystal-masked-unknown.v0";
pub const N_TREES: usize = 50;
pub const MAX_DEPTH: usize = 8;
pub const MIN_SAMPLES: usize = 3;
pub const RNG_SEED: u64 = 42;
pub const GROUPED_OOF_FOLDS: usize = 5;
pub const RANDOM_FEATURES_PER_SPLIT: usize = 4;
pub const THRESHOLD_CANDIDATE_DIVISOR: usize = 10;
pub const RNG_LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
pub const RNG_LCG_INCREMENT: u64 = 1_442_695_040_888_963_407;
const FRACTION_SCALE: f64 = 1_000_000_000.0;

/// Fully explicit semantic recipe for the composition-only learned model.
///
/// Exact implementation/toolchain identity remains a separate qualification
/// concern; this object freezes the model semantics that must not silently drift
/// while retaining the same model identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositionOnlyModelRecipe {
    pub model_id: String,
    pub model_version: String,
    pub feature_schema: String,
    pub composition_identity: String,
    pub crystal_feature_policy: String,
    pub duplicate_composition_policy: String,
    pub group_target_aggregation: String,
    pub residual_baseline: String,
    pub n_trees: usize,
    pub max_depth: usize,
    pub min_samples: usize,
    pub rng_seed: u64,
    pub rng_algorithm: String,
    pub rng_lcg_multiplier: u64,
    pub rng_lcg_increment: u64,
    pub bootstrap_policy: String,
    pub random_features_per_split: usize,
    pub threshold_candidate_divisor: usize,
    pub threshold_policy: String,
    pub grouped_oof_folds: usize,
    pub grouped_oof_assignment: String,
}

pub fn model_recipe() -> CompositionOnlyModelRecipe {
    CompositionOnlyModelRecipe {
        model_id: MODEL_ID.into(),
        model_version: MODEL_VERSION.into(),
        feature_schema: FEATURE_SCHEMA.into(),
        composition_identity: "normalized-atomic-fraction-quantized-1e-9".into(),
        crystal_feature_policy: "masked-to-Unknown-for-all-training-and-inference-samples".into(),
        duplicate_composition_policy:
            "one-equal-weight-training-sample-per-composition-group".into(),
        group_target_aggregation: "arithmetic-mean-observed-gap-per-composition-group".into(),
        residual_baseline: "symthaea.bandgap.electronegativity-baseline".into(),
        n_trees: N_TREES,
        max_depth: MAX_DEPTH,
        min_samples: MIN_SAMPLES,
        rng_seed: RNG_SEED,
        rng_algorithm: "wrapping-lcg64-then-upper31-bits-modulo-bound".into(),
        rng_lcg_multiplier: RNG_LCG_MULTIPLIER,
        rng_lcg_increment: RNG_LCG_INCREMENT,
        bootstrap_policy: "n-draws-with-replacement-per-tree".into(),
        random_features_per_split: RANDOM_FEATURES_PER_SPLIT,
        threshold_candidate_divisor: THRESHOLD_CANDIDATE_DIVISOR,
        threshold_policy:
            "sorted-unique-feature-values-step-by-max(len/divisor,1)-including-first".into(),
        grouped_oof_folds: GROUPED_OOF_FOLDS,
        grouped_oof_assignment:
            "canonical-BTreeMap-composition-group-order-modulo-fold-count".into(),
    }
}

/// Phase/structure-insensitive composition identity used by model training,
/// inference canonicalization, and grouped out-of-fold partitioning.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CompositionGroupId(pub Vec<(u8, u64)>);

/// One deterministic grouped out-of-fold prediction over the curated Symthaea
/// training table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompositionOnlyOofPrediction {
    pub training_index: usize,
    pub formula: String,
    pub composition_group: CompositionGroupId,
    pub fold: usize,
    pub group_size: usize,
    /// Mean observed band gap across all curated polymorph/material rows with
    /// this exact composition-only identity.
    pub group_mean_gap_ev: f64,
    /// Standard deviation of observed gaps inside this composition group. This
    /// is descriptive ambiguity, not model uncertainty.
    pub group_observed_std_ev: f64,
    pub predicted_gap_ev: f64,
    pub observed_gap_ev: f64,
    /// Inter-tree standard deviation only; it is not calibrated total error.
    pub uncertainty_ev: f64,
}

impl CompositionOnlyOofPrediction {
    pub fn error_ev(&self) -> f64 {
        self.predicted_gap_ev - self.observed_gap_ev
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositionOnlyError {
    InvalidComposition(String),
}

impl fmt::Display for CompositionOnlyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidComposition(message) => write!(f, "invalid composition: {message}"),
        }
    }
}

impl Error for CompositionOnlyError {}

#[derive(Debug, Clone)]
struct GroupTrainingSample {
    id: CompositionGroupId,
    composition: Vec<(u8, f64)>,
    member_indices: Vec<usize>,
    mean_gap_ev: f64,
    observed_std_ev: f64,
}

/// Deterministic composition-only residual Random Forest.
///
/// Exact composition duplicates/polymorphs are collapsed to one equal-weight
/// group-mean training target because this representation has no information
/// with which to distinguish their structures.
pub struct CompositionOnlyBandgapPredictor {
    forest: RandomForest,
}

impl CompositionOnlyBandgapPredictor {
    pub fn new() -> Self {
        Self::train_from_entries(&load_training_data())
    }

    fn train_from_entries(data: &[MaterialEntry]) -> Self {
        let groups = aggregate_training_entries(data);
        Self::train_from_groups(&groups)
    }

    fn train_from_groups(groups: &[GroupTrainingSample]) -> Self {
        let mut features = Vec::with_capacity(groups.len());
        let mut residuals = Vec::with_capacity(groups.len());
        for group in groups {
            let baseline = electronegativity_bandgap(&group.composition);
            features.push(masked_features(&group.composition));
            residuals.push(group.mean_gap_ev - baseline);
        }
        let forest = RandomForest::train(
            &features,
            &residuals,
            N_TREES,
            MAX_DEPTH,
            MIN_SAMPLES,
            RNG_SEED,
        );
        Self { forest }
    }

    /// Predict from composition only. There is deliberately no crystal-system
    /// parameter in this API. Arbitrary positive stoichiometric scaling is
    /// canonicalized before feature extraction.
    pub fn predict(
        &self,
        composition: &[(u8, f64)],
    ) -> Result<BandgapPrediction, CompositionOnlyError> {
        let group = composition_group_id(composition)?;
        let canonical = canonical_composition(&group)?;
        let features = masked_features(&canonical);
        let baseline = electronegativity_bandgap(&canonical);
        let (correction, uncertainty) = self.forest.predict(&features);
        Ok(BandgapPrediction {
            bandgap: (baseline + correction).max(0.0),
            baseline,
            ml_correction: correction,
            uncertainty,
        })
    }

    /// Deterministic grouped 5-fold out-of-fold predictions over the current
    /// curated training table.
    ///
    /// Every exact normalized composition group is assigned wholly to one fold,
    /// so polymorphs sharing a composition cannot leak that composition between
    /// train and test partitions.
    pub fn grouped_oof_predictions() -> Vec<CompositionOnlyOofPrediction> {
        grouped_oof_from_entries(&load_training_data())
    }

    pub fn grouped_cross_validate() -> (f64, f64) {
        let predictions = Self::grouped_oof_predictions();
        let count = predictions.len() as f64;
        let mae = predictions
            .iter()
            .map(|prediction| prediction.error_ev().abs())
            .sum::<f64>()
            / count;
        let rmse = (predictions
            .iter()
            .map(|prediction| prediction.error_ev().powi(2))
            .sum::<f64>()
            / count)
            .sqrt();
        (mae, rmse)
    }
}

impl Default for CompositionOnlyBandgapPredictor {
    fn default() -> Self {
        Self::new()
    }
}

/// Canonical normalized composition group identity used by training and OOF.
pub fn composition_group_id(
    composition: &[(u8, f64)],
) -> Result<CompositionGroupId, CompositionOnlyError> {
    if composition.is_empty() {
        return Err(CompositionOnlyError::InvalidComposition(
            "composition cannot be empty".into(),
        ));
    }
    let mut amounts: BTreeMap<u8, f64> = BTreeMap::new();
    for &(atomic_number, amount) in composition {
        if atomic_number == 0 || !amount.is_finite() || amount <= 0.0 {
            return Err(CompositionOnlyError::InvalidComposition(format!(
                "element Z={atomic_number} has non-finite or non-positive amount {amount}"
            )));
        }
        *amounts.entry(atomic_number).or_insert(0.0) += amount;
    }
    let total: f64 = amounts.values().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(CompositionOnlyError::InvalidComposition(
            "composition total must be finite and positive".into(),
        ));
    }

    let mut fingerprint = Vec::with_capacity(amounts.len());
    for (atomic_number, amount) in amounts {
        let quantized = ((amount / total) * FRACTION_SCALE).round();
        if !quantized.is_finite() || quantized <= 0.0 || quantized > u64::MAX as f64 {
            return Err(CompositionOnlyError::InvalidComposition(format!(
                "element Z={atomic_number} cannot be represented in composition-group space"
            )));
        }
        fingerprint.push((atomic_number, quantized as u64));
    }
    Ok(CompositionGroupId(fingerprint))
}

fn canonical_composition(
    group: &CompositionGroupId,
) -> Result<Vec<(u8, f64)>, CompositionOnlyError> {
    if group.0.is_empty() {
        return Err(CompositionOnlyError::InvalidComposition(
            "composition group cannot be empty".into(),
        ));
    }
    let total: u128 = group.0.iter().map(|(_, amount)| u128::from(*amount)).sum();
    if total == 0 {
        return Err(CompositionOnlyError::InvalidComposition(
            "composition group has zero total".into(),
        ));
    }
    let mut previous = None;
    let mut composition = Vec::with_capacity(group.0.len());
    for &(atomic_number, amount) in &group.0 {
        if atomic_number == 0 || amount == 0 || previous.is_some_and(|z| z >= atomic_number) {
            return Err(CompositionOnlyError::InvalidComposition(
                "composition group entries must be positive and strictly ordered".into(),
            ));
        }
        previous = Some(atomic_number);
        composition.push((atomic_number, amount as f64 / total as f64));
    }
    Ok(composition)
}

fn masked_features(composition: &[(u8, f64)]) -> [f64; N_FEATURES] {
    extract_features(composition, &CrystalSystem::Unknown)
}

fn aggregate_training_entries(data: &[MaterialEntry]) -> Vec<GroupTrainingSample> {
    let mut members: BTreeMap<CompositionGroupId, Vec<usize>> = BTreeMap::new();
    for (index, entry) in data.iter().enumerate() {
        let group = composition_group_id(&entry.composition)
            .expect("curated band-gap training compositions must be valid");
        members.entry(group).or_default().push(index);
    }

    members
        .into_iter()
        .map(|(id, member_indices)| {
            let composition = canonical_composition(&id)
                .expect("curated composition group identity must be canonical");
            let mean_gap_ev = member_indices
                .iter()
                .map(|&index| data[index].bandgap_exp())
                .sum::<f64>()
                / member_indices.len() as f64;
            let observed_std_ev = (member_indices
                .iter()
                .map(|&index| (data[index].bandgap_exp() - mean_gap_ev).powi(2))
                .sum::<f64>()
                / member_indices.len() as f64)
                .sqrt();
            GroupTrainingSample {
                id,
                composition,
                member_indices,
                mean_gap_ev,
                observed_std_ev,
            }
        })
        .collect()
}

fn grouped_oof_from_entries(data: &[MaterialEntry]) -> Vec<CompositionOnlyOofPrediction> {
    let groups = aggregate_training_entries(data);
    assert!(
        groups.len() >= GROUPED_OOF_FOLDS,
        "composition-only grouped OOF requires at least one composition group per fold"
    );

    let mut output = Vec::with_capacity(data.len());
    for fold in 0..GROUPED_OOF_FOLDS {
        let train_groups: Vec<_> = groups
            .iter()
            .enumerate()
            .filter(|(group_ordinal, _)| group_ordinal % GROUPED_OOF_FOLDS != fold)
            .map(|(_, group)| group.clone())
            .collect();
        let predictor = CompositionOnlyBandgapPredictor::train_from_groups(&train_groups);

        for (group_ordinal, group) in groups.iter().enumerate() {
            if group_ordinal % GROUPED_OOF_FOLDS != fold {
                continue;
            }
            let prediction = predictor
                .predict(&group.composition)
                .expect("canonical held-out composition must be valid");
            for &training_index in &group.member_indices {
                let entry = &data[training_index];
                output.push(CompositionOnlyOofPrediction {
                    training_index,
                    formula: entry.formula.to_owned(),
                    composition_group: group.id.clone(),
                    fold,
                    group_size: group.member_indices.len(),
                    group_mean_gap_ev: group.mean_gap_ev,
                    group_observed_std_ev: group.observed_std_ev,
                    predicted_gap_ev: prediction.bandgap,
                    observed_gap_ev: entry.bandgap_exp(),
                    uncertainty_ev: prediction.uncertainty,
                });
            }
        }
    }
    output.sort_by_key(|prediction| prediction.training_index);
    assert_eq!(output.len(), data.len());
    output
}

#[derive(Debug, Clone)]
enum TreeNode {
    Leaf(f64),
    Split {
        feature: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl TreeNode {
    fn predict(&self, features: &[f64; N_FEATURES]) -> f64 {
        match self {
            Self::Leaf(value) => *value,
            Self::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if features[*feature] <= *threshold {
                    left.predict(features)
                } else {
                    right.predict(features)
                }
            }
        }
    }
}

struct RandomForest {
    trees: Vec<TreeNode>,
}

impl RandomForest {
    fn train(
        features: &[[f64; N_FEATURES]],
        targets: &[f64],
        n_trees: usize,
        max_depth: usize,
        min_samples: usize,
        seed: u64,
    ) -> Self {
        assert!(!features.is_empty());
        assert_eq!(features.len(), targets.len());
        let n = features.len();
        let mut trees = Vec::with_capacity(n_trees);
        let mut rng_state = seed;

        for _ in 0..n_trees {
            let indices: Vec<usize> = (0..n)
                .map(|_| {
                    rng_state = rng_state
                        .wrapping_mul(RNG_LCG_MULTIPLIER)
                        .wrapping_add(RNG_LCG_INCREMENT);
                    (rng_state >> 33) as usize % n
                })
                .collect();
            let boot_features: Vec<_> = indices.iter().map(|&index| features[index]).collect();
            let boot_targets: Vec<_> = indices.iter().map(|&index| targets[index]).collect();
            trees.push(Self::build_tree(
                &boot_features,
                &boot_targets,
                0,
                max_depth,
                min_samples,
                &mut rng_state,
            ));
        }
        Self { trees }
    }

    fn build_tree(
        features: &[[f64; N_FEATURES]],
        targets: &[f64],
        depth: usize,
        max_depth: usize,
        min_samples: usize,
        rng: &mut u64,
    ) -> TreeNode {
        let n = features.len();
        let mean_target = targets.iter().sum::<f64>() / n as f64;
        if n <= min_samples || depth >= max_depth {
            return TreeNode::Leaf(mean_target);
        }
        let variance = targets
            .iter()
            .map(|target| (target - mean_target).powi(2))
            .sum::<f64>()
            / n as f64;
        if variance < 1e-10 {
            return TreeNode::Leaf(mean_target);
        }

        let mut best_feature = 0usize;
        let mut best_threshold = 0.0;
        let mut best_score = f64::INFINITY;
        for _ in 0..RANDOM_FEATURES_PER_SPLIT {
            *rng = rng
                .wrapping_mul(RNG_LCG_MULTIPLIER)
                .wrapping_add(RNG_LCG_INCREMENT);
            let feature_index = (*rng >> 33) as usize % N_FEATURES;
            let mut values: Vec<f64> = features.iter().map(|row| row[feature_index]).collect();
            values.sort_by(f64::total_cmp);
            values.dedup();
            let step = (values.len() / THRESHOLD_CANDIDATE_DIVISOR).max(1);
            for value_index in (0..values.len()).step_by(step) {
                let threshold = values[value_index];
                let (left_sum, left_sq, left_n, right_sum, right_sq, right_n) = features
                    .iter()
                    .zip(targets)
                    .fold(
                        (0.0f64, 0.0f64, 0usize, 0.0f64, 0.0f64, 0usize),
                        |(ls, lsq, ln, rs, rsq, rn), (row, &target)| {
                            if row[feature_index] <= threshold {
                                (ls + target, lsq + target * target, ln + 1, rs, rsq, rn)
                            } else {
                                (ls, lsq, ln, rs + target, rsq + target * target, rn + 1)
                            }
                        },
                    );
                if left_n < min_samples || right_n < min_samples {
                    continue;
                }
                let left_var = left_sq / left_n as f64 - (left_sum / left_n as f64).powi(2);
                let right_var = right_sq / right_n as f64 - (right_sum / right_n as f64).powi(2);
                let score =
                    left_n as f64 * left_var.max(0.0) + right_n as f64 * right_var.max(0.0);
                if score < best_score {
                    best_score = score;
                    best_feature = feature_index;
                    best_threshold = threshold;
                }
            }
        }
        if best_score == f64::INFINITY {
            return TreeNode::Leaf(mean_target);
        }

        let (left_features, left_targets): (Vec<_>, Vec<_>) = features
            .iter()
            .zip(targets)
            .filter(|(row, _)| row[best_feature] <= best_threshold)
            .map(|(row, &target)| (*row, target))
            .unzip();
        let (right_features, right_targets): (Vec<_>, Vec<_>) = features
            .iter()
            .zip(targets)
            .filter(|(row, _)| row[best_feature] > best_threshold)
            .map(|(row, &target)| (*row, target))
            .unzip();
        if left_features.is_empty() || right_features.is_empty() {
            return TreeNode::Leaf(mean_target);
        }

        TreeNode::Split {
            feature: best_feature,
            threshold: best_threshold,
            left: Box::new(Self::build_tree(
                &left_features,
                &left_targets,
                depth + 1,
                max_depth,
                min_samples,
                rng,
            )),
            right: Box::new(Self::build_tree(
                &right_features,
                &right_targets,
                depth + 1,
                max_depth,
                min_samples,
                rng,
            )),
        }
    }

    fn predict(&self, features: &[f64; N_FEATURES]) -> (f64, f64) {
        let predictions: Vec<f64> = self
            .trees
            .iter()
            .map(|tree| tree.predict(features))
            .collect();
        let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let variance = predictions
            .iter()
            .map(|prediction| (prediction - mean).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;
        (mean, variance.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_prediction_requires_no_crystal_system() {
        let predictor = CompositionOnlyBandgapPredictor::new();
        let prediction = predictor.predict(&[(31, 0.5), (33, 0.5)]).unwrap();
        assert!(prediction.bandgap.is_finite() && prediction.bandgap >= 0.0);
        assert!(prediction.uncertainty.is_finite() && prediction.uncertainty >= 0.0);
    }

    #[test]
    fn scaled_stoichiometry_has_identical_prediction() {
        let predictor = CompositionOnlyBandgapPredictor::new();
        let first = predictor.predict(&[(31, 1.0), (33, 1.0)]).unwrap();
        let second = predictor.predict(&[(31, 0.5), (33, 0.5)]).unwrap();
        assert_eq!(first.bandgap.to_bits(), second.bandgap.to_bits());
        assert_eq!(first.uncertainty.to_bits(), second.uncertainty.to_bits());
    }

    #[test]
    fn masked_feature_is_constant_unknown_ordinal() {
        let features = masked_features(&[(31, 0.5), (33, 0.5)]);
        assert_eq!(features[16], CrystalSystem::Unknown.ordinal() as f64);
    }

    #[test]
    fn changing_only_training_crystal_labels_cannot_change_model() {
        let original = load_training_data();
        let mut relabeled = original.clone();
        for entry in &mut relabeled {
            entry.crystal_system = CrystalSystem::Triclinic;
        }
        let first = CompositionOnlyBandgapPredictor::train_from_entries(&original);
        let second = CompositionOnlyBandgapPredictor::train_from_entries(&relabeled);
        for composition in [
            vec![(14, 1.0)],
            vec![(31, 0.5), (33, 0.5)],
            vec![(30, 0.5), (8, 0.5)],
        ] {
            let left = first.predict(&composition).unwrap();
            let right = second.predict(&composition).unwrap();
            assert_eq!(left.bandgap.to_bits(), right.bandgap.to_bits());
            assert_eq!(left.uncertainty.to_bits(), right.uncertainty.to_bits());
        }
    }

    #[test]
    fn duplicate_compositions_collapse_to_one_equal_weight_training_sample() {
        let data = load_training_data();
        let groups = aggregate_training_entries(&data);
        assert!(groups.len() < data.len());
        let sic = groups
            .iter()
            .find(|group| {
                group.member_indices.len() >= 3
                    && group
                        .member_indices
                        .iter()
                        .all(|&index| data[index].formula.starts_with("SiC-"))
            })
            .expect("SiC polymorph group should be collapsed");
        assert!(sic.observed_std_ev > 0.0);
    }

    #[test]
    fn grouped_oof_covers_every_training_entry_once() {
        let data = load_training_data();
        let predictions = CompositionOnlyBandgapPredictor::grouped_oof_predictions();
        assert_eq!(predictions.len(), data.len());
        for (index, prediction) in predictions.iter().enumerate() {
            assert_eq!(prediction.training_index, index);
            assert!(prediction.fold < GROUPED_OOF_FOLDS);
            assert!(prediction.predicted_gap_ev.is_finite());
            assert!(prediction.uncertainty_ev.is_finite());
            assert!(prediction.group_observed_std_ev.is_finite());
        }
    }

    #[test]
    fn identical_composition_groups_never_split_across_folds() {
        let predictions = CompositionOnlyBandgapPredictor::grouped_oof_predictions();
        let mut group_to_fold = BTreeMap::new();
        for prediction in predictions {
            let fold = prediction.fold;
            if let Some(previous) = group_to_fold.insert(prediction.composition_group, fold) {
                assert_eq!(previous, fold);
            }
        }
    }

    #[test]
    fn sic_polymorphs_are_held_out_together() {
        let predictions = CompositionOnlyBandgapPredictor::grouped_oof_predictions();
        let sic: Vec<_> = predictions
            .iter()
            .filter(|prediction| prediction.formula.starts_with("SiC-"))
            .collect();
        assert!(sic.len() >= 3);
        let fold = sic[0].fold;
        let group = &sic[0].composition_group;
        assert!(sic.iter().all(|prediction| {
            prediction.fold == fold && &prediction.composition_group == group
        }));
    }

    #[test]
    fn fixed_recipe_is_bitwise_deterministic() {
        let first = CompositionOnlyBandgapPredictor::new();
        let second = CompositionOnlyBandgapPredictor::new();
        let composition = [(31, 0.5), (33, 0.5)];
        let left = first.predict(&composition).unwrap();
        let right = second.predict(&composition).unwrap();
        assert_eq!(left.bandgap.to_bits(), right.bandgap.to_bits());
        assert_eq!(left.baseline.to_bits(), right.baseline.to_bits());
        assert_eq!(left.ml_correction.to_bits(), right.ml_correction.to_bits());
        assert_eq!(left.uncertainty.to_bits(), right.uncertainty.to_bits());
    }

    #[test]
    fn grouped_cv_is_finite() {
        let (mae, rmse) = CompositionOnlyBandgapPredictor::grouped_cross_validate();
        assert!(mae.is_finite() && mae > 0.0);
        assert!(rmse.is_finite() && rmse > 0.0);
    }

    #[test]
    fn recipe_freezes_model_hyperparameters_and_group_policy() {
        let recipe = model_recipe();
        assert_eq!(recipe.model_id, MODEL_ID);
        assert_eq!(recipe.model_version, MODEL_VERSION);
        assert_eq!(recipe.n_trees, 50);
        assert_eq!(recipe.max_depth, 8);
        assert_eq!(recipe.min_samples, 3);
        assert_eq!(recipe.rng_seed, 42);
        assert_eq!(recipe.rng_lcg_multiplier, RNG_LCG_MULTIPLIER);
        assert_eq!(recipe.rng_lcg_increment, RNG_LCG_INCREMENT);
        assert_eq!(recipe.random_features_per_split, 4);
        assert_eq!(recipe.threshold_candidate_divisor, 10);
        assert_eq!(recipe.grouped_oof_folds, 5);
        assert!(recipe.duplicate_composition_policy.contains("group"));
        assert!(recipe.bootstrap_policy.contains("replacement"));
        assert!(recipe.grouped_oof_assignment.contains("BTreeMap"));
    }
}
