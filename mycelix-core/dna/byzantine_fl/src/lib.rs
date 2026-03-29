// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// TODO: HDK 0.4→0.6 upgrade complete. Old `entry_defs` callback replaced with
// `#[hdk_entry_helper]` + `#[hdk_entry_types]` macros. Entry types are not yet
// used by zome functions (submit_gradient stores nothing in DHT), so this is
// forward-compatible scaffolding for when DHT storage is implemented.

use hdk::prelude::*;

/// Gradient entry for federated learning
#[hdk_entry_helper]
#[derive(Clone)]
pub struct GradientEntry {
    pub worker_id: String,
    pub round: u32,
    pub values: Vec<f64>,
    pub timestamp: i64,
}

/// Aggregated model entry
#[hdk_entry_helper]
#[derive(Clone)]
pub struct AggregatedModelEntry {
    pub round: u32,
    pub model: Vec<f64>,
    pub detected_byzantine: Vec<String>,
}

/// Plain gradient (not a DHT entry — used for function inputs/outputs)
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Gradient {
    pub worker_id: String,
    pub round: u32,
    pub values: Vec<f64>,
    pub timestamp: i64,
}

/// Aggregated model (not a DHT entry — used for function outputs)
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct AggregatedModel {
    pub round: u32,
    pub model: Vec<f64>,
    pub detected_byzantine: Vec<String>,
}

/// Byzantine detection result
#[derive(Serialize, Deserialize, Debug)]
pub struct ByzantineDetection {
    pub suspects: Vec<String>,
    pub confidence: f64,
}

/// Krum aggregation input
#[derive(Serialize, Deserialize, Debug)]
pub struct KrumInput {
    pub gradients: Vec<Gradient>,
    pub round: u32,
}

/// Entry type definitions (HDK 0.6 pattern)
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    GradientEntry(GradientEntry),
    AggregatedModelEntry(AggregatedModelEntry),
}

/// Link types (required by HDK 0.6 even if unused)
#[hdk_link_types]
pub enum LinkTypes {
    GradientToRound,
}

#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    Ok(InitCallbackResult::Pass)
}

/// Submit a gradient for a training round
#[hdk_extern]
pub fn submit_gradient(gradient: Gradient) -> ExternResult<String> {
    // For now, just store and return a hash string
    let _gradient_json = serde_json::to_string(&gradient)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

    debug!(
        "Gradient submitted: worker={}, round={}",
        gradient.worker_id, gradient.round
    );

    // Return a mock hash for now
    let hash = format!("gradient_{}_{}", gradient.round, gradient.worker_id);
    Ok(hash)
}

/// Get gradients for a round (simplified)
#[hdk_extern]
pub fn get_round_gradients(round: u32) -> ExternResult<Vec<Gradient>> {
    debug!("Getting gradients for round {}", round);

    // For now, return empty list
    Ok(Vec::new())
}

/// Detect Byzantine nodes
#[hdk_extern]
pub fn detect_byzantine(gradients: Vec<Gradient>) -> ExternResult<ByzantineDetection> {
    let mut suspects = Vec::new();

    // Simple outlier detection
    if gradients.len() > 2 {
        // Calculate mean
        let dim = gradients.first().map(|g| g.values.len()).unwrap_or(0);
        if dim == 0 {
            return Ok(ByzantineDetection {
                suspects,
                confidence: 0.0,
            });
        }

        let mut mean = vec![0.0; dim];
        for gradient in &gradients {
            for (i, val) in gradient.values.iter().enumerate() {
                if i < dim {
                    mean[i] += val;
                }
            }
        }

        for val in &mut mean {
            *val /= gradients.len() as f64;
        }

        // Check for outliers
        for gradient in &gradients {
            let mut total_diff = 0.0;
            for (i, val) in gradient.values.iter().enumerate() {
                if i < dim {
                    let diff = (val - mean[i]).abs();
                    total_diff += diff;
                }
            }

            // If total difference is too large, mark as Byzantine
            if total_diff > mean.iter().sum::<f64>() * 3.0 {
                suspects.push(gradient.worker_id.clone());
            }
        }
    }

    let confidence = if suspects.is_empty() { 1.0 } else { 0.8 };

    Ok(ByzantineDetection {
        suspects,
        confidence,
    })
}

/// Krum aggregation
#[hdk_extern]
pub fn krum_aggregate(input: KrumInput) -> ExternResult<AggregatedModel> {
    let gradients = input.gradients;
    let round = input.round;

    if gradients.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest("No gradients".into())));
    }

    // For simplicity, just average non-Byzantine gradients
    let detection = detect_byzantine(gradients.clone())?;

    let clean_gradients: Vec<&Gradient> = gradients
        .iter()
        .filter(|g| !detection.suspects.contains(&g.worker_id))
        .collect();

    if clean_gradients.is_empty() {
        // If all are Byzantine, use all anyway
        let first = &gradients[0];
        return Ok(AggregatedModel {
            round,
            model: first.values.clone(),
            detected_byzantine: detection.suspects,
        });
    }

    // Average the clean gradients
    let dim = clean_gradients[0].values.len();
    let mut model = vec![0.0; dim];

    for gradient in &clean_gradients {
        for (i, val) in gradient.values.iter().enumerate() {
            if i < dim {
                model[i] += val;
            }
        }
    }

    for val in &mut model {
        *val /= clean_gradients.len() as f64;
    }

    Ok(AggregatedModel {
        round,
        model,
        detected_byzantine: detection.suspects,
    })
}

/// Training metrics input
#[derive(Serialize, Deserialize, Debug)]
pub struct TrainingMetrics {
    pub round: u32,
    pub accuracy: f64,
    pub loss: f64,
}

/// Store training metrics (simplified)
#[hdk_extern]
pub fn store_metrics(metrics: TrainingMetrics) -> ExternResult<()> {
    debug!(
        "Storing metrics: round={}, accuracy={}, loss={}",
        metrics.round, metrics.accuracy, metrics.loss
    );
    Ok(())
}
