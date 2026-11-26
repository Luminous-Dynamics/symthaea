// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! GRAPE online adapter parity primitives.
//! This module is deliberately separate from evidence-gated delivery semantics.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

pub const GRAPE_ADAPTER_PARITY_SCHEMA_V1: &str = "luminous.semi-bench.grape-adapter-parity.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrapeAdapterKindV1 {
    Nlms,
    Rls,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrapeAdapterPolicyV1 {
    pub schema: String,
    pub adapter_kind: GrapeAdapterKindV1,
    pub nlms_learning_rate: f64,
    pub rls_forgetting_factor: f64,
    pub rls_ridge_initial: f64,
    pub nlms_norm_floor: f64,
    pub positive_definite_tolerance: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrapeFeatureAdapterStateV1 {
    pub kind: GrapeAdapterKindV1,
    pub weights: Vec<f64>,
    pub covariance: Option<Vec<Vec<f64>>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrapeContextStatsV1 {
    pub mean: f64,
    pub count: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrapeHierarchicalContextStateV1 {
    pub ema_alpha: f64,
    pub minimum_support: u64,
    pub global_mean: f64,
    pub global_count: u64,
    pub equipment: BTreeMap<String, GrapeContextStatsV1>,
    pub chamber: BTreeMap<String, GrapeContextStatsV1>,
    pub equipment_chamber: BTreeMap<String, GrapeContextStatsV1>,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct GrapeAdapterParityErrorV1(pub String);

fn finite(value: f64, field: &str) -> Result<(), GrapeAdapterParityErrorV1> {
    if !value.is_finite() {
        return Err(GrapeAdapterParityErrorV1(format!("{field} must be finite")));
    }
    Ok(())
}

fn validate_policy(policy: &GrapeAdapterPolicyV1) -> Result<(), GrapeAdapterParityErrorV1> {
    if policy.schema != GRAPE_ADAPTER_PARITY_SCHEMA_V1
        || !policy.nlms_learning_rate.is_finite()
        || policy.nlms_learning_rate < 0.0
        || !policy.rls_forgetting_factor.is_finite()
        || !(0.0..=1.0).contains(&policy.rls_forgetting_factor)
        || policy.rls_forgetting_factor == 0.0
        || !policy.rls_ridge_initial.is_finite()
        || policy.rls_ridge_initial <= 0.0
        || !policy.nlms_norm_floor.is_finite()
        || policy.nlms_norm_floor <= 0.0
        || !policy.positive_definite_tolerance.is_finite()
        || policy.positive_definite_tolerance <= 0.0
    {
        return Err(GrapeAdapterParityErrorV1(
            "invalid GRAPE adapter policy".into(),
        ));
    }
    Ok(())
}

fn validate_covariance(
    matrix: &[Vec<f64>],
    n: usize,
    tolerance: f64,
) -> Result<(), GrapeAdapterParityErrorV1> {
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        return Err(GrapeAdapterParityErrorV1(
            "RLS covariance shape mismatch".into(),
        ));
    }
    for row in matrix {
        for value in row {
            finite(*value, "RLS covariance")?;
        }
    }
    for i in 0..n {
        for j in 0..n {
            if (matrix[i][j] - matrix[j][i]).abs() > tolerance {
                return Err(GrapeAdapterParityErrorV1(
                    "RLS covariance is not symmetric".into(),
                ));
            }
        }
    }
    let mut lower = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut value = matrix[i][j];
            for k in 0..j {
                value -= lower[i][k] * lower[j][k];
            }
            if i == j {
                if value <= tolerance {
                    return Err(GrapeAdapterParityErrorV1(
                        "RLS covariance is not positive definite".into(),
                    ));
                }
                lower[i][j] = value.sqrt();
            } else {
                lower[i][j] = value / lower[j][j];
            }
        }
    }
    Ok(())
}

impl GrapeFeatureAdapterStateV1 {
    pub fn predict(&self, x: &[f64]) -> Result<f64, GrapeAdapterParityErrorV1> {
        if x.len() != self.weights.len() || x.is_empty() || x.iter().any(|v| !v.is_finite()) {
            return Err(GrapeAdapterParityErrorV1(
                "feature adapter input is invalid".into(),
            ));
        }
        let result = self.weights.iter().zip(x).map(|(w, v)| w * v).sum::<f64>();
        finite(result, "feature adapter prediction")?;
        Ok(result)
    }

    pub fn update(
        &mut self,
        policy: &GrapeAdapterPolicyV1,
        x: &[f64],
        error: f64,
    ) -> Result<(), GrapeAdapterParityErrorV1> {
        validate_policy(policy)?;
        finite(error, "adapter error")?;
        if self.kind != policy.adapter_kind
            || x.len() != self.weights.len()
            || x.is_empty()
            || x.iter().any(|v| !v.is_finite())
        {
            return Err(GrapeAdapterParityErrorV1(
                "adapter kind or feature dimension mismatch".into(),
            ));
        }
        let n = self.weights.len();
        let mut next = self.clone();
        match self.kind {
            GrapeAdapterKindV1::Nlms => {
                let denom = x.iter().map(|v| v * v).sum::<f64>() + policy.nlms_norm_floor;
                finite(denom, "NLMS denominator")?;
                for (w, v) in next.weights.iter_mut().zip(x) {
                    *w += policy.nlms_learning_rate * error * v / denom;
                    finite(*w, "NLMS weight")?;
                }
            }
            GrapeAdapterKindV1::Rls => {
                let p = self
                    .covariance
                    .as_ref()
                    .ok_or_else(|| GrapeAdapterParityErrorV1("RLS covariance missing".into()))?;
                validate_covariance(p, n, policy.positive_definite_tolerance)?;
                let mut px = vec![0.0; n];
                let mut x_p = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        px[i] += p[i][j] * x[j];
                        x_p[j] += x[i] * p[i][j];
                    }
                }
                let denom = policy.rls_forgetting_factor
                    + x.iter().zip(&px).map(|(a, b)| a * b).sum::<f64>();
                finite(denom, "RLS denominator")?;
                if denom <= policy.positive_definite_tolerance {
                    return Err(GrapeAdapterParityErrorV1(
                        "RLS denominator is singular".into(),
                    ));
                }
                let gain: Vec<f64> = px.iter().map(|v| v / denom).collect();
                for (w, k) in next.weights.iter_mut().zip(&gain) {
                    *w += k * error;
                    finite(*w, "RLS weight")?;
                }
                let mut updated = vec![vec![0.0; n]; n];
                for i in 0..n {
                    for j in 0..n {
                        updated[i][j] = (p[i][j] - gain[i] * x_p[j]) / policy.rls_forgetting_factor;
                    }
                }
                for i in 0..n {
                    for j in (i + 1)..n {
                        let v = 0.5 * (updated[i][j] + updated[j][i]);
                        updated[i][j] = v;
                        updated[j][i] = v;
                    }
                }
                validate_covariance(&updated, n, policy.positive_definite_tolerance)?;
                next.covariance = Some(updated);
            }
        }
        *self = next;
        Ok(())
    }
}

fn reliability(count: u64, minimum: u64) -> f64 {
    if minimum == 0 {
        1.0
    } else {
        let n = count as f64;
        n / (n + minimum as f64)
    }
}

impl GrapeHierarchicalContextStateV1 {
    pub fn new(alpha: f64, minimum_support: u64) -> Result<Self, GrapeAdapterParityErrorV1> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(GrapeAdapterParityErrorV1(
                "context EMA alpha invalid".into(),
            ));
        }
        Ok(Self {
            ema_alpha: alpha,
            minimum_support,
            global_mean: 0.0,
            global_count: 0,
            equipment: BTreeMap::new(),
            chamber: BTreeMap::new(),
            equipment_chamber: BTreeMap::new(),
        })
    }

    fn update_bucket(
        bucket: &mut BTreeMap<String, GrapeContextStatsV1>,
        key: &str,
        residual: f64,
        alpha: f64,
    ) -> Result<(), GrapeAdapterParityErrorV1> {
        if let Some(stats) = bucket.get_mut(key) {
            let count = stats
                .count
                .checked_add(1)
                .ok_or_else(|| GrapeAdapterParityErrorV1("context count overflow".into()))?;
            let mean = (1.0 - alpha) * stats.mean + alpha * residual;
            finite(mean, "context mean")?;
            stats.mean = mean;
            stats.count = count;
        } else {
            bucket.insert(
                key.into(),
                GrapeContextStatsV1 {
                    mean: residual,
                    count: 1,
                },
            );
        }
        Ok(())
    }

    pub fn update(
        &mut self,
        key: Option<&str>,
        residual: f64,
    ) -> Result<(), GrapeAdapterParityErrorV1> {
        finite(residual, "context residual")?;
        if key.is_some_and(|v| v.trim().is_empty()) {
            return Err(GrapeAdapterParityErrorV1(
                "context key cannot be empty".into(),
            ));
        }
        let mut next = self.clone();
        next.global_count = next
            .global_count
            .checked_add(1)
            .ok_or_else(|| GrapeAdapterParityErrorV1("global count overflow".into()))?;
        next.global_mean = (1.0 - next.ema_alpha) * next.global_mean + next.ema_alpha * residual;
        finite(next.global_mean, "global mean")?;
        if let Some(key) = key {
            let (eqp, chm) = key.rsplit_once('_').map_or((key, ""), |(a, b)| (a, b));
            if !eqp.is_empty() {
                Self::update_bucket(&mut next.equipment, eqp, residual, next.ema_alpha)?;
            }
            if !chm.is_empty() {
                Self::update_bucket(&mut next.chamber, chm, residual, next.ema_alpha)?;
            }
            Self::update_bucket(&mut next.equipment_chamber, key, residual, next.ema_alpha)?;
        }
        *self = next;
        Ok(())
    }

    pub fn correction(&self, key: Option<&str>) -> Result<f64, GrapeAdapterParityErrorV1> {
        if key.is_some_and(|v| v.trim().is_empty()) {
            return Err(GrapeAdapterParityErrorV1("context key cannot be empty".into()));
        }
        let mut values = vec![self.global_mean];
        let mut weights = vec![reliability(self.global_count, self.minimum_support)];
        if let Some(key) = key {
            let (eqp, chm) = key.rsplit_once('_').map_or((key, ""), |(a,b)| (a,b));
            if let Some(s) = self.equipment.get(eqp) {
                values.push(s.mean);
                weights.push(reliability(s.count, self.minimum_support));
            }
            if let Some(s) = self.chamber.get(chm).filter(|_| !chm.is_empty()) {
                values.push(s.mean);
                weights.push(reliability(s.count, self.minimum_support));
            }
            if let Some(s) = self.equipment_chamber.get(key) {
                values.push(s.mean);
                weights.push(reliability(s.count, self.minimum_support));
            }
        }
        let total = weights.iter().sum::<f64>();
        if total <= 1.0e-12 {
            return Ok(0.0);
        }
        let result = values.iter().zip(weights).map(|(v, w)| v * w).sum::<f64>() / total;
        finite(result, "context correction")?;
        Ok(result)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrapeTemporalResidualStateV1 {
    pub ema_alpha: f64,
    pub activation_threshold: f64,
    pub residual_ema: f64,
}

impl GrapeTemporalResidualStateV1 {
    pub fn new(
        ema_alpha: f64,
        activation_threshold: f64,
    ) -> Result<Self, GrapeAdapterParityErrorV1> {
        finite(ema_alpha, "temporal EMA alpha")?;
        finite(activation_threshold, "temporal activation threshold")?;
        if !(0.0..=1.0).contains(&ema_alpha) || activation_threshold < 0.0 {
            return Err(GrapeAdapterParityErrorV1(
                "temporal policy is out of range".into(),
            ));
        }
        Ok(Self {
            ema_alpha,
            activation_threshold,
            residual_ema: 0.0,
        })
    }

    pub fn calibrate(
        &mut self,
        training_residuals: &[f64],
        percentile: f64,
    ) -> Result<(), GrapeAdapterParityErrorV1> {
        finite(percentile, "temporal percentile")?;
        if !(0.0..=100.0).contains(&percentile) || training_residuals.iter().any(|v| !v.is_finite())
        {
            return Err(GrapeAdapterParityErrorV1(
                "invalid temporal calibration input".into(),
            ));
        }
        if training_residuals.is_empty() {
            self.activation_threshold = self.activation_threshold.max(1.0e-6);
            return Ok(());
        }
        let mut values: Vec<f64> = training_residuals.iter().map(|v| v.abs()).collect();
        values.sort_by(f64::total_cmp);
        let pos = (values.len() - 1) as f64 * percentile / 100.0;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let fraction = pos - lo as f64;
        self.activation_threshold =
            (values[lo] * (1.0 - fraction) + values[hi] * fraction).max(1.0e-6);
        Ok(())
    }

    pub fn correction(&self) -> f64 {
        if self.residual_ema.abs() < self.activation_threshold {
            0.0
        } else {
            self.residual_ema
        }
    }

    pub fn update(&mut self, residual: f64) -> Result<(), GrapeAdapterParityErrorV1> {
        finite(residual, "temporal residual")?;
        let next = (1.0 - self.ema_alpha) * self.residual_ema + self.ema_alpha * residual;
        finite(next, "temporal residual EMA")?;
        self.residual_ema = next;
        Ok(())
    }
}

pub fn create_grape_feature_adapter_v1(
    kind: GrapeAdapterKindV1,
    dimension: usize,
    policy: &GrapeAdapterPolicyV1,
) -> Result<GrapeFeatureAdapterStateV1, GrapeAdapterParityErrorV1> {
    validate_policy(policy)?;
    if dimension == 0 || kind != policy.adapter_kind {
        return Err(GrapeAdapterParityErrorV1(
            "adapter dimension or kind is invalid".into(),
        ));
    }
    let covariance = if kind == GrapeAdapterKindV1::Rls {
        let diagonal = policy.rls_ridge_initial;
        Some(
            (0..dimension)
                .map(|row| {
                    (0..dimension)
                        .map(|col| if row == col { diagonal } else { 0.0 })
                        .collect()
                })
                .collect(),
        )
    } else {
        None
    };
    Ok(GrapeFeatureAdapterStateV1 {
        kind,
        weights: vec![0.0; dimension],
        covariance,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrapeRuntimePolicyV1 {
    pub adapter_policy: GrapeAdapterPolicyV1,
    pub adapter_feature_count: usize,
    pub context_ema_alpha: f64,
    pub context_minimum_support: u64,
    pub temporal_ema_alpha: f64,
    pub temporal_activation_percentile: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrapePredictionDecompositionV1 {
    pub global_model_prediction: f64,
    pub selected_feature_values: Vec<f64>,
    pub adapter_correction: f64,
    pub context_correction: f64,
    pub unclipped_base_prediction: f64,
    pub clipped_base_prediction: f64,
    pub temporal_correction: f64,
    pub final_prediction: f64,
    pub context_key: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrapeAdapterRuntimeStateV1 {
    pub adapter: GrapeFeatureAdapterStateV1,
    pub context: GrapeHierarchicalContextStateV1,
    pub temporal: GrapeTemporalResidualStateV1,
    pub accepted_update_count: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrapeAdapterRuntimeV1 {
    pub policy: GrapeRuntimePolicyV1,
    pub selected_feature_indices: Vec<usize>,
    pub base_label_min: f64,
    pub base_label_max: f64,
    pub label_min: f64,
    pub label_max: f64,
    pub state: GrapeAdapterRuntimeStateV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrapeAdapterUpdateReceiptV1 {
    pub state_before_sha256: String,
    pub state_after_sha256: String,
    pub accepted_update_count: u64,
    pub feature_error: f64,
    pub temporal_error: f64,
    pub context_residual: Option<f64>,
    pub state_mutated: bool,
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, GrapeAdapterParityErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        GrapeAdapterParityErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn percentile_linear(values: &[f64], percentile: f64) -> Result<f64, GrapeAdapterParityErrorV1> {
    if values.is_empty() || !percentile.is_finite() || !(0.0..=100.0).contains(&percentile)
        || values.iter().any(|value| !value.is_finite())
    {
        return Err(GrapeAdapterParityErrorV1("percentile input is invalid".into()));
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = (sorted.len() - 1) as f64 * percentile / 100.0;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    Ok(sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction)
}

/// Selects top adapter features by absolute Pearson correlation with training
/// residuals. Ties are deterministically broken by ascending source index.
pub fn select_grape_adapter_features_v1(
    rows: &[Vec<f64>],
    training_residuals: &[f64],
    count: usize,
) -> Result<Vec<usize>, GrapeAdapterParityErrorV1> {
    if rows.is_empty() || rows.len() != training_residuals.len() || count == 0 {
        return Err(GrapeAdapterParityErrorV1("feature selection shape or count is invalid".into()));
    }
    let dimension = rows[0].len();
    if dimension == 0 || rows.iter().any(|row| row.len() != dimension
        || row.iter().any(|value| !value.is_finite()))
        || training_residuals.iter().any(|value| !value.is_finite())
    {
        return Err(GrapeAdapterParityErrorV1("feature selection contains invalid values".into()));
    }
    let take = count.min(dimension);
    let residual_mean = training_residuals.iter().sum::<f64>() / training_residuals.len() as f64;
    let residual_variance = training_residuals
        .iter()
        .map(|value| (value - residual_mean).powi(2))
        .sum::<f64>() / training_residuals.len() as f64;
    let residual_std = residual_variance.sqrt();
    if residual_std < 1.0e-8 {
        return Ok((0..take).collect());
    }

    let mut correlations = Vec::new();
    for column in 0..dimension {
        let mean = rows.iter().map(|row| row[column]).sum::<f64>() / rows.len() as f64;
        let variance = rows.iter().map(|row| (row[column] - mean).powi(2)).sum::<f64>()
            / rows.len() as f64;
        let feature_std = variance.sqrt();
        if feature_std <= 1.0e-8 {
            continue;
        }
        let covariance = rows
            .iter()
            .zip(training_residuals)
            .map(|(row, residual)| (row[column] - mean) * (residual - residual_mean))
            .sum::<f64>() / rows.len() as f64;
        correlations.push((column, (covariance / (feature_std * residual_std)).abs()));
    }
    if correlations.is_empty() {
        return Ok((0..take).collect());
    }
    correlations.sort_by(|(left_index, left), (right_index, right)| {
        right.total_cmp(left).then_with(|| left_index.cmp(right_index))
    });
    Ok(correlations.into_iter().take(take).map(|(index, _)| index).collect())
}

impl GrapeAdapterRuntimeV1 {
    /// Initializes adapter metadata and clipping/temporal bounds from train-only
    /// features, targets and predictions of the frozen global Ridge model.
    pub fn initialize(
        policy: GrapeRuntimePolicyV1,
        training_features: &[Vec<f64>],
        training_targets: &[f64],
        global_training_predictions: &[f64],
    ) -> Result<Self, GrapeAdapterParityErrorV1> {
        validate_policy(&policy.adapter_policy)?;
        if training_features.is_empty()
            || training_features.len() != training_targets.len()
            || training_features.len() != global_training_predictions.len()
            || training_targets.iter().any(|value| !value.is_finite())
            || global_training_predictions.iter().any(|value| !value.is_finite())
        {
            return Err(GrapeAdapterParityErrorV1("GRAPE training initialization shape is invalid".into()));
        }
        if !policy.context_ema_alpha.is_finite()
            || !(0.0..=1.0).contains(&policy.context_ema_alpha)
            || !policy.temporal_ema_alpha.is_finite()
            || !(0.0..=1.0).contains(&policy.temporal_ema_alpha)
            || !policy.temporal_activation_percentile.is_finite()
            || !(0.0..=100.0).contains(&policy.temporal_activation_percentile)
            || policy.adapter_feature_count == 0
        {
            return Err(GrapeAdapterParityErrorV1("GRAPE runtime policy is invalid".into()));
        }

        let residuals: Vec<f64> = training_targets.iter()
            .zip(global_training_predictions)
            .map(|(target, prediction)| target - prediction)
            .collect();
        let selected_feature_indices =
            select_grape_adapter_features_v1(training_features, &residuals, policy.adapter_feature_count)?;
        let mut adapter_policy = policy.adapter_policy.clone();
        let target_mean = training_targets.iter().sum::<f64>() / training_targets.len() as f64;
        let target_std = (training_targets.iter()
            .map(|value| (value - target_mean).powi(2))
            .sum::<f64>() / training_targets.len() as f64).sqrt();
        let mut context_minimum_support = policy.context_minimum_support;
        if target_std < 0.3 {
            adapter_policy.nlms_learning_rate = adapter_policy.nlms_learning_rate.max(0.02);
            context_minimum_support = context_minimum_support.max(25);
        }

        let low = percentile_linear(training_targets, 1.0)?;
        let high = percentile_linear(training_targets, 99.0)?;
        let base_label_min = low - 1.0;
        let base_label_max = high + 1.0;
        let absolute_residuals: Vec<f64> = residuals.iter().map(|value| value.abs()).collect();
        let temporal_threshold = percentile_linear(
            &absolute_residuals,
            policy.temporal_activation_percentile,
        )?.max(1.0e-6);
        let temporal_pad = 2.0_f64.max(6.0 * temporal_threshold);
        let label_min = low - temporal_pad;
        let label_max = high + temporal_pad;

        let adapter = create_grape_feature_adapter_v1(
            adapter_policy.adapter_kind,
            selected_feature_indices.len(),
            &adapter_policy,
        )?;
        let context = GrapeHierarchicalContextStateV1::new(
            policy.context_ema_alpha,
            context_minimum_support,
        )?;
        let mut temporal = GrapeTemporalResidualStateV1::new(
            policy.temporal_ema_alpha,
            temporal_threshold,
        )?;
        temporal.calibrate(&residuals, policy.temporal_activation_percentile)?;

        let mut effective_policy = policy;
        effective_policy.adapter_policy = adapter_policy;
        effective_policy.context_minimum_support = context_minimum_support;
        Ok(Self {
            policy: effective_policy,
            selected_feature_indices,
            base_label_min,
            base_label_max,
            label_min,
            label_max,
            state: GrapeAdapterRuntimeStateV1 {
                adapter,
                context,
                temporal,
                accepted_update_count: 0,
            },
        })
    }

    /// Makes a prediction from an externally supplied frozen global Ridge output.
    /// This computes the exact GRAPE decomposition but does not mutate runtime state.
    pub fn predict(
        &self,
        global_model_prediction: f64,
        features: &[f64],
        context_key: Option<&str>,
    ) -> Result<GrapePredictionDecompositionV1, GrapeAdapterParityErrorV1> {
        finite(global_model_prediction, "global model prediction")?;
        if features.is_empty() || features.iter().any(|value| !value.is_finite())
            || self.selected_feature_indices.iter().any(|index| *index >= features.len())
        {
            return Err(GrapeAdapterParityErrorV1("GRAPE prediction feature vector is invalid".into()));
        }
        let selected_feature_values: Vec<f64> = self.selected_feature_indices
            .iter().map(|index| features[*index]).collect();
        let adapter_correction = self.state.adapter.predict(&selected_feature_values)?;
        let context_correction = self.state.context.correction(context_key)?;
        let unclipped_base_prediction =
            global_model_prediction + adapter_correction + context_correction;
        let clipped_base_prediction =
            unclipped_base_prediction.clamp(self.base_label_min, self.base_label_max);
        let temporal_correction = self.state.temporal.correction();
        let final_prediction =
            (clipped_base_prediction + temporal_correction).clamp(self.label_min, self.label_max);
        finite(final_prediction, "GRAPE final prediction")?;
        Ok(GrapePredictionDecompositionV1 {
            global_model_prediction,
            selected_feature_values,
            adapter_correction,
            context_correction,
            unclipped_base_prediction,
            clipped_base_prediction,
            temporal_correction,
            final_prediction,
            context_key: context_key.map(str::to_string),
        })
    }

    /// Applies the pinned GRAPE update equations to a frozen prediction record.
    /// Mutations are transactional; callers must separately apply K1 evidence
    /// qualification before invoking this reference math primitive.
    pub fn update_from_prediction(
        &mut self,
        prediction: &GrapePredictionDecompositionV1,
        target: f64,
    ) -> Result<GrapeAdapterUpdateReceiptV1, GrapeAdapterParityErrorV1> {
        finite(target, "GRAPE target")?;
        let before = json_sha256(&self.state)?;
        let mut next = self.state.clone();
        let temporal_target = target.clamp(self.label_min, self.label_max);
        let base_target = temporal_target.clamp(self.base_label_min, self.base_label_max);
        let temporal_error = temporal_target - prediction.clipped_base_prediction;
        let feature_error = base_target - prediction.clipped_base_prediction;

        next.temporal.update(temporal_error)?;
        next.adapter.update(
            &self.policy.adapter_policy,
            &prediction.selected_feature_values,
            feature_error,
        )?;
        let updated_adapter_correction =
            next.adapter.predict(&prediction.selected_feature_values)?;
        let context_residual = base_target - (
            prediction.global_model_prediction + updated_adapter_correction
        );
        if prediction.context_key.is_some() {
            next.context.update(prediction.context_key.as_deref(), context_residual)?;
        }
        next.accepted_update_count = next.accepted_update_count.checked_add(1)
            .ok_or_else(|| GrapeAdapterParityErrorV1("GRAPE update counter overflow".into()))?;
        let after = json_sha256(&next)?;
        let receipt = GrapeAdapterUpdateReceiptV1 {
            state_before_sha256: before,
            state_after_sha256: after,
            accepted_update_count: next.accepted_update_count,
            feature_error,
            temporal_error,
            context_residual: prediction.context_key.as_ref().map(|_| context_residual),
            state_mutated: true,
        };
        self.state = next;
        Ok(receipt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(kind: GrapeAdapterKindV1) -> GrapeAdapterPolicyV1 {
        GrapeAdapterPolicyV1 {
            schema: GRAPE_ADAPTER_PARITY_SCHEMA_V1.into(),
            adapter_kind: kind,
            nlms_learning_rate: 0.01,
            rls_forgetting_factor: 0.99,
            rls_ridge_initial: 1.0,
            nlms_norm_floor: 1.0e-3,
            positive_definite_tolerance: 1.0e-12,
        }
    }

    #[test]
    fn rls_scalar_update_matches_hand_computed_reference() {
        let p = policy(GrapeAdapterKindV1::Rls);
        let mut adapter = create_grape_feature_adapter_v1(GrapeAdapterKindV1::Rls, 1, &p).unwrap();
        adapter.update(&p, &[2.0], 0.5).unwrap();
        let expected_gain = 2.0 / 4.99;
        let expected_p = (1.0 - expected_gain * 2.0) / 0.99;
        assert!((adapter.weights[0] - expected_gain * 0.5).abs() < 1.0e-12);
        assert!((adapter.covariance.as_ref().unwrap()[0][0] - expected_p).abs() < 1.0e-12);
    }

    #[test]
    fn rls_initial_covariance_matches_upstream_ridge_init_multiplier() {
        let mut p = policy(GrapeAdapterKindV1::Rls);
        p.rls_ridge_initial = 2.5;
        let adapter = create_grape_feature_adapter_v1(GrapeAdapterKindV1::Rls, 3, &p).unwrap();
        assert_eq!(
            adapter.covariance.unwrap(),
            vec![
                vec![2.5, 0.0, 0.0],
                vec![0.0, 2.5, 0.0],
                vec![0.0, 0.0, 2.5]
            ]
        );
    }

    #[test]
    fn rls_covariance_stays_symmetric_positive_definite_under_updates() {
        let p = policy(GrapeAdapterKindV1::Rls);
        let mut adapter = create_grape_feature_adapter_v1(GrapeAdapterKindV1::Rls, 3, &p).unwrap();
        for i in 0..200 {
            let x = [
                ((i * 7 % 19) as f64 - 9.0) / 5.0,
                ((i * 11 % 23) as f64 - 11.0) / 7.0,
                ((i * 13 % 29) as f64 - 14.0) / 9.0,
            ];
            adapter
                .update(&p, &x, ((i % 17) as f64 - 8.0) / 4.0)
                .unwrap();
        }
        validate_covariance(
            adapter.covariance.as_ref().unwrap(),
            3,
            p.positive_definite_tolerance,
        )
        .unwrap();
    }

    #[test]
    fn nlms_scalar_update_matches_upstream_equation() {
        let p = policy(GrapeAdapterKindV1::Nlms);
        let mut adapter = create_grape_feature_adapter_v1(GrapeAdapterKindV1::Nlms, 1, &p).unwrap();
        adapter.update(&p, &[2.0], 0.5).unwrap();
        assert!((adapter.weights[0] - (0.01 * 0.5 * 2.0 / 4.001)).abs() < 1.0e-12);
    }

    #[test]
    fn failed_rls_update_is_atomic() {
        let p = policy(GrapeAdapterKindV1::Rls);
        let mut adapter = create_grape_feature_adapter_v1(GrapeAdapterKindV1::Rls, 2, &p).unwrap();
        let before = adapter.clone();
        assert!(adapter.update(&p, &[f64::NAN, 1.0], 0.5).is_err());
        assert_eq!(adapter, before);
    }

    #[test]
    fn hierarchical_context_matches_four_level_reliability_weighting() {
        let mut context = GrapeHierarchicalContextStateV1::new(0.5, 2).unwrap();
        context.update(Some("eqpA_ch1"), 2.0).unwrap();
        context.update(Some("eqpA_ch2"), 6.0).unwrap();
        context.update(Some("eqpB_ch1"), 10.0).unwrap();
        let actual = context.correction(Some("eqpA_ch1")).unwrap();

        let weighted = [
            (context.global_mean, reliability(context.global_count, 2)),
            (context.equipment["eqpA"].mean, reliability(2, 2)),
            (context.chamber["ch1"].mean, reliability(2, 2)),
            (
                context.equipment_chamber["eqpA_ch1"].mean,
                reliability(1, 2),
            ),
        ];
        let expected = weighted.iter().map(|(v, w)| v * w).sum::<f64>()
            / weighted.iter().map(|(_, w)| w).sum::<f64>();
        assert!((actual - expected).abs() < 1.0e-12);
    }

    #[test]
    fn unseen_context_falls_back_to_global_not_another_local_bucket() {
        let mut context = GrapeHierarchicalContextStateV1::new(0.1, 10).unwrap();
        context.update(Some("eqpA_ch1"), 8.0).unwrap();
        assert_eq!(
            context.correction(Some("eqpB_ch9")).unwrap(),
            context.global_mean
        );
        assert!(!context.equipment.contains_key("eqpB"));
        assert!(!context.equipment_chamber.contains_key("eqpB_ch9"));
    }

    #[test]
    fn context_count_overflow_does_not_mutate_hierarchy() {
        let mut context = GrapeHierarchicalContextStateV1::new(0.5, 2).unwrap();
        context.global_count = u64::MAX;
        let before = context.clone();
        assert!(context.update(Some("eqpA_ch1"), 2.0).is_err());
        assert_eq!(context, before);
    }

    fn runtime_policy(kind: GrapeAdapterKindV1) -> GrapeRuntimePolicyV1 {
        GrapeRuntimePolicyV1 {
            adapter_policy: policy(kind),
            adapter_feature_count: 2,
            context_ema_alpha: 0.05,
            context_minimum_support: 10,
            temporal_ema_alpha: 0.5,
            temporal_activation_percentile: 96.0,
        }
    }

    fn runtime_training_data() -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        (
            vec![
                vec![1.0, 1.0, 4.0],
                vec![2.0, -1.0, 3.0],
                vec![3.0, 1.0, 2.0],
                vec![4.0, -1.0, 1.0],
            ],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0.0, 0.0, 0.0, 0.0],
        )
    }

    #[test]
    fn residual_correlation_selection_matches_known_feature_order() {
        let (features, targets, predictions) = runtime_training_data();
        let residuals: Vec<f64> = targets.iter().zip(&predictions)
            .map(|(target, prediction)| target - prediction)
            .collect();
        assert_eq!(
            select_grape_adapter_features_v1(&features, &residuals, 2).unwrap(),
            vec![0, 2]
        );
    }

    #[test]
    fn runtime_initialization_matches_training_bounds_and_temporal_percentile() {
        let (features, targets, predictions) = runtime_training_data();
        let runtime = GrapeAdapterRuntimeV1::initialize(
            runtime_policy(GrapeAdapterKindV1::Nlms),
            &features,
            &targets,
            &predictions,
        )
        .unwrap();
        assert_eq!(runtime.selected_feature_indices, vec![0, 2]);
        assert!((runtime.base_label_min - 0.03).abs() < 1.0e-12);
        assert!((runtime.base_label_max - 4.97).abs() < 1.0e-12);
        assert!((runtime.state.temporal.activation_threshold - 3.88).abs() < 1.0e-12);
        assert!((runtime.label_min - (1.03 - 23.28)).abs() < 1.0e-12);
        assert!((runtime.label_max - (3.97 + 23.28)).abs() < 1.0e-12);
    }

    #[test]
    fn small_target_variance_applies_upstream_nlms_and_context_minimum_adjustments() {
        let features = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.5, 0.5],
            vec![0.25, 0.75],
        ];
        let targets = vec![1.0, 1.1, 0.9, 1.05];
        let predictions = vec![1.0, 1.0, 1.0, 1.0];
        let runtime = GrapeAdapterRuntimeV1::initialize(
            runtime_policy(GrapeAdapterKindV1::Nlms),
            &features,
            &targets,
            &predictions,
        )
        .unwrap();
        assert_eq!(runtime.policy.context_minimum_support, 25);
        assert_eq!(runtime.state.context.minimum_support, 25);
        assert_eq!(runtime.policy.adapter_policy.nlms_learning_rate, 0.02);
    }

    #[test]
    fn runtime_prediction_matches_base_clipping_and_temporal_decomposition() {
        let (features, targets, predictions) = runtime_training_data();
        let mut runtime = GrapeAdapterRuntimeV1::initialize(
            runtime_policy(GrapeAdapterKindV1::Nlms),
            &features,
            &targets,
            &predictions,
        )
        .unwrap();
        let prediction = runtime.predict(100.0, &[0.0, 0.0, 0.0], Some("eqpA_ch1")).unwrap();
        assert!((prediction.unclipped_base_prediction - 100.0).abs() < 1.0e-12);
        assert!((prediction.clipped_base_prediction - 4.97).abs() < 1.0e-12);
        assert_eq!(prediction.temporal_correction, 0.0);
        assert!((prediction.final_prediction - 4.97).abs() < 1.0e-12);
        assert_eq!(runtime.state.accepted_update_count, 0);
    }

    #[test]
    fn runtime_update_uses_grape_base_error_then_updates_context_residual() {
        let (features, targets, predictions) = runtime_training_data();
        let mut runtime = GrapeAdapterRuntimeV1::initialize(
            runtime_policy(GrapeAdapterKindV1::Nlms),
            &features,
            &targets,
            &predictions,
        )
        .unwrap();
        let prediction = runtime.predict(2.0, &[1.0, 0.0, 0.0], Some("eqpA_ch1")).unwrap();
        let update = runtime.update_from_prediction(&prediction, 3.0).unwrap();
        assert_eq!(update.feature_error, 1.0);
        assert_eq!(update.temporal_error, 1.0);
        assert!(update.context_residual.is_some());
        assert_eq!(update.accepted_update_count, 1);
        assert!(update.state_mutated);
        assert!((runtime.state.adapter.weights[0] - (0.01 / 1.001)).abs() < 1.0e-12);
        assert_eq!(runtime.state.context.global_count, 1);
        assert!((runtime.state.temporal.residual_ema - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn no_context_key_does_not_update_hierarchical_context_but_updates_adapters() {
        let (features, targets, predictions) = runtime_training_data();
        let mut runtime = GrapeAdapterRuntimeV1::initialize(
            runtime_policy(GrapeAdapterKindV1::Nlms),
            &features,
            &targets,
            &predictions,
        )
        .unwrap();
        let prediction = runtime.predict(2.0, &[1.0, 0.0, 0.0], None).unwrap();
        let receipt = runtime.update_from_prediction(&prediction, 3.0).unwrap();
        assert_eq!(receipt.context_residual, None);
        assert_eq!(runtime.state.context.global_count, 0);
        assert_eq!(runtime.state.accepted_update_count, 1);
    }

    #[test]
    fn runtime_update_is_atomic_when_context_counter_overflows() {
        let (features, targets, predictions) = runtime_training_data();
        let mut runtime = GrapeAdapterRuntimeV1::initialize(
            runtime_policy(GrapeAdapterKindV1::Nlms),
            &features,
            &targets,
            &predictions,
        )
        .unwrap();
        let prediction = runtime.predict(2.0, &[1.0, 0.0, 0.0], Some("eqpA_ch1")).unwrap();
        runtime.state.context.global_count = u64::MAX;
        let before = runtime.state.clone();
        assert!(runtime.update_from_prediction(&prediction, 3.0).is_err());
        assert_eq!(runtime.state, before);
    }

    #[test]
    fn runtime_replay_is_deterministic_for_same_training_and_updates() {
        fn run() -> String {
            let (features, targets, predictions) = runtime_training_data();
            let mut runtime = GrapeAdapterRuntimeV1::initialize(
                runtime_policy(GrapeAdapterKindV1::Rls),
                &features,
                &targets,
                &predictions,
            )
            .unwrap();
            for target in [2.5, 1.5, 3.5] {
                let prediction = runtime.predict(2.0, &[1.0, 0.0, 0.0], Some("eqpA_ch1")).unwrap();
                runtime.update_from_prediction(&prediction, target).unwrap();
            }
            json_sha256(&runtime.state).unwrap()
        }
        assert_eq!(run(), run());
    }

    #[test]
    fn temporal_threshold_uses_linear_percentile_and_ema_activation() {
        let mut temporal = GrapeTemporalResidualStateV1::new(0.5, 0.1).unwrap();
        temporal.calibrate(&[-1.0, 2.0, -3.0, 4.0], 50.0).unwrap();
        assert!((temporal.activation_threshold - 2.5).abs() < 1.0e-12);
        temporal.update(6.0).unwrap();
        assert_eq!(temporal.residual_ema, 3.0);
        assert_eq!(temporal.correction(), 3.0);
    }
}
