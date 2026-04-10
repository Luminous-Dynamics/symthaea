// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdk::prelude::*;

// Helper to apply Sigmoid normalization to reputation scores
// S(x) = 1 / (1 + exp(-k * (x - x0)))
// k = steepness (e.g., 10.0 for sharp transition)
// x0 = threshold midpoint (e.g., 0.5)

pub fn apply_sigmoid_normalization(score: f64) -> f64 {
    let k = 10.0;
    let x0 = 0.5;
    1.0 / (1.0 + (-(k * (score - x0))).exp())
}

/// Compute recursive reputation for an agent via the identity bridge's
/// exponential-decay-weighted aggregation.
///
/// Delegates to `identity_bridge::get_reputation_score` which aggregates
/// across all hApps with a 30-day half-life weighting.  Falls back to
/// sigmoid-normalized 0.0 (≈0.007) when the bridge is unreachable.
pub fn calculate_recursive_reputation(agent_b64: String) -> ExternResult<f64> {
    // Build the DID that the identity bridge expects.
    let did = format!("did:mycelix:{}", agent_b64);

    match call(
        CallTargetCell::Local,
        ZomeName::new("identity_bridge"),
        FunctionName::new("get_reputation_score"),
        None,
        did,
    ) {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            let score: f64 = extern_io
                .decode()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(
                    format!("Failed to decode reputation score: {}", e)
                )))?;
            Ok(score.clamp(0.0, 1.0))
        }
        Ok(_) | Err(_) => {
            // Identity bridge unreachable within this DNA — fall back to 0.0
            // which the caller normalizes via sigmoid.
            Ok(0.0)
        }
    }
}
