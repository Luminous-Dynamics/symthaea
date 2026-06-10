// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phi-gated attention utilities for CfC networks.
//!
//! Uses IIT Phi values to weight attention across multiple inputs,
//! allowing higher-consciousness streams to dominate network output.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Configuration for Phi-gated attention in CfC networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiGatedConfig {
    /// Temperature for softmax (lower = sharper attention)
    pub temperature: f32,

    /// Scale factor for Phi values (learnable)
    pub scale: f32,

    /// Bias for Phi values (learnable)
    pub bias: f32,

    /// Minimum attention weight
    pub min_attention: f32,
}

impl Default for PhiGatedConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            scale: 1.0,
            bias: 0.0,
            min_attention: 0.0,
        }
    }
}

impl PhiGatedConfig {
    /// Create config with sharp attention (low temperature)
    pub fn sharp() -> Self {
        Self {
            temperature: 0.1,
            ..Default::default()
        }
    }

    /// Create config with soft attention (high temperature)
    pub fn soft() -> Self {
        Self {
            temperature: 5.0,
            ..Default::default()
        }
    }
}

/// Compute attention weights from Phi values using softmax with temperature.
///
/// Higher Phi values receive higher attention weights.
pub fn compute_phi_attention_weights(phi_values: &[f64], config: &PhiGatedConfig) -> Vec<f32> {
    if phi_values.is_empty() {
        return vec![];
    }

    // Transform Phi values
    let transformed: Vec<f32> = phi_values
        .iter()
        .map(|&p| config.scale * p as f32 + config.bias)
        .collect();

    // Apply softmax with temperature
    let temp = config.temperature.max(1e-10);
    let max_val = transformed
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let exp_values: Vec<f32> = transformed
        .iter()
        .map(|&v| ((v - max_val) / temp).exp())
        .collect();

    let sum: f32 = exp_values.iter().sum();

    if sum < 1e-10 {
        // Uniform fallback
        let n = phi_values.len() as f32;
        return vec![1.0 / n; phi_values.len()];
    }

    let mut weights: Vec<f32> = exp_values.iter().map(|&e| e / sum).collect();

    // Apply minimum attention floor
    if config.min_attention > 0.0 {
        let floor = config.min_attention;
        let n = weights.len() as f32;

        for w in weights.iter_mut() {
            *w = *w * (1.0 - floor * n) + floor;
        }

        // Renormalize
        let new_sum: f32 = weights.iter().sum();
        if new_sum > 1e-10 {
            for w in weights.iter_mut() {
                *w /= new_sum;
            }
        }
    }

    weights
}

/// Compute weighted bundle of ndarray arrays
pub(crate) fn weighted_array_bundle(arrays: &[Array1<f32>], weights: &[f32]) -> Array1<f32> {
    if arrays.is_empty() || weights.is_empty() {
        return Array1::zeros(0);
    }

    let dim = arrays[0].len();
    let mut result = Array1::zeros(dim);

    for (arr, &w) in arrays.iter().zip(weights.iter()) {
        for i in 0..dim.min(arr.len()) {
            result[i] += w * arr[i];
        }
    }

    result
}
