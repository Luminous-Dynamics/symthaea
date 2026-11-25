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
pub enum GrapeAdapterKindV1 { Nlms, Rls }

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
pub struct GrapeContextStatsV1 { pub mean: f64, pub count: u64 }

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
        return Err(GrapeAdapterParityErrorV1("invalid GRAPE adapter policy".into()));
    }
    Ok(())
}

fn validate_covariance(
    matrix: &[Vec<f64>],
    n: usize,
    tolerance: f64,
) -> Result<(), GrapeAdapterParityErrorV1> {
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        return Err(GrapeAdapterParityErrorV1("RLS covariance shape mismatch".into()));
    }
    for row in matrix {
        for value in row { finite(*value, "RLS covariance")?; }
    }
    for i in 0..n {
        for j in 0..n {
            if (matrix[i][j] - matrix[j][i]).abs() > tolerance {
                return Err(GrapeAdapterParityErrorV1("RLS covariance is not symmetric".into()));
            }
        }
    }
    let mut lower = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut value = matrix[i][j];
            for k in 0..j { value -= lower[i][k] * lower[j][k]; }
            if i == j {
                if value <= tolerance {
                    return Err(GrapeAdapterParityErrorV1("RLS covariance is not positive definite".into()));
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
            return Err(GrapeAdapterParityErrorV1("feature adapter input is invalid".into()));
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
        if self.kind != policy.adapter_kind || x.len() != self.weights.len() || x.is_empty()
            || x.iter().any(|v| !v.is_finite())
        {
            return Err(GrapeAdapterParityErrorV1("adapter kind or feature dimension mismatch".into()));
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
                let p = self.covariance.as_ref().ok_or_else(|| GrapeAdapterParityErrorV1("RLS covariance missing".into()))?;
                validate_covariance(p, n, policy.positive_definite_tolerance)?;
                let mut px = vec![0.0; n];
                let mut x_p = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        px[i] += p[i][j] * x[j];
                        x_p[j] += x[i] * p[i][j];
                    }
                }
                let denom = policy.rls_forgetting_factor + x.iter().zip(&px).map(|(a,b)| a*b).sum::<f64>();
                finite(denom, "RLS denominator")?;
                if denom <= policy.positive_definite_tolerance {
                    return Err(GrapeAdapterParityErrorV1("RLS denominator is singular".into()));
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
                    for j in (i+1)..n {
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
    if minimum == 0 { 1.0 } else {
        let n = count as f64;
        n / (n + minimum as f64)
    }
}

impl GrapeHierarchicalContextStateV1 {
    pub fn new(alpha: f64, minimum_support: u64) -> Result<Self, GrapeAdapterParityErrorV1> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(GrapeAdapterParityErrorV1("context EMA alpha invalid".into()));
        }
        Ok(Self { ema_alpha: alpha, minimum_support, global_mean: 0.0, global_count: 0,
            equipment: BTreeMap::new(), chamber: BTreeMap::new(), equipment_chamber: BTreeMap::new() })
    }

    fn update_bucket(
        bucket: &mut BTreeMap<String, GrapeContextStatsV1>,
        key: &str,
        residual: f64,
        alpha: f64,
    ) -> Result<(), GrapeAdapterParityErrorV1> {
        if let Some(stats) = bucket.get_mut(key) {
            let count = stats.count.checked_add(1).ok_or_else(|| GrapeAdapterParityErrorV1("context count overflow".into()))?;
            let mean = (1.0 - alpha) * stats.mean + alpha * residual;
            finite(mean, "context mean")?;
            stats.mean = mean;
            stats.count = count;
        } else {
            bucket.insert(key.into(), GrapeContextStatsV1 { mean: residual, count: 1 });
        }
        Ok(())
    }

    pub fn update(&mut self, key: Option<&str>, residual: f64) -> Result<(), GrapeAdapterParityErrorV1> {
        finite(residual, "context residual")?;
        if key.is_some_and(|v| v.trim().is_empty()) {
            return Err(GrapeAdapterParityErrorV1("context key cannot be empty".into()));
        }
        let mut next = self.clone();
        next.global_count = next.global_count.checked_add(1).ok_or_else(|| GrapeAdapterParityErrorV1("global count overflow".into()))?;
        next.global_mean = (1.0 - next.ema_alpha) * next.global_mean + next.ema_alpha * residual;
        finite(next.global_mean, "global mean")?;
        if let Some(key) = key {
            let (eqp, chm) = key.rsplit_once('_').map_or((key, ""), |(a,b)| (a,b));
            if !eqp.is_empty() { Self::update_bucket(&mut next.equipment, eqp, residual, next.ema_alpha)?; }
            if !chm.is_empty() { Self::update_bucket(&mut next.chamber, chm, residual, next.ema_alpha)?; }
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
            if let Some(s) = self.equipment.get(eqp) { values.push(s.mean); weights.push(reliability(s.count, self.minimum_support)); }
            if let Some(s) = self.chamber.get(chm).filter(|_| !chm.is_empty()) { values.push(s.mean); weights.push(reliability(s.count, self.minimum_support)); }
            if let Some(s) = self.equipment_chamber.get(key) { values.push(s.mean); weights.push(reliability(s.count, self.minimum_support)); }
        }
        let total = weights.iter().sum::<f64>();
        if total <= 1.0e-12 { return Ok(0.0); }
        let result = values.iter().zip(weights).map(|(v,w)| v*w).sum::<f64>() / total;
        finite(result, "context correction")?;
        Ok(result)
    }
}
