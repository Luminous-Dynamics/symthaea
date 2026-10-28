// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Research adapter for the fail-closed PyPhi comparison.
//!
//! Reads one explicit state-by-node TPM and state from stdin, validates the input,
//! and returns Symthaea's current full-system-mechanism bipartition score.
//! This output is a candidate score for empirical comparison with PyPhi IIT_3_0
//! system integrated information; it is not asserted to be formally equivalent.

use serde::Deserialize;
use serde_json::json;
use std::error::Error;
use std::io::{self, Read};

use symthaea::hdc::iit_exact::{system_phi, TransitionProbabilityMatrix};

const INPUT_SCHEMA: &str = "symthaea-iit3-tpm-input-v1";
const OUTPUT_SCHEMA: &str = "symthaea-iit3-tpm-output-v1";
const FORMALISM_PROFILE: &str = "IIT_3_0";
const METRIC_ID: &str = "symthaea_iit3_full_system_mechanism_mip_candidate_v1";
const MAX_NODES: usize = 8;

#[derive(Debug, Deserialize)]
struct Input {
    schema_version: String,
    model_sha256: String,
    formalism_profile: String,
    n_nodes: usize,
    state: Vec<u8>,
    tpm: Vec<Vec<f64>>,
}

fn validate_hex_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut bytes = Vec::new();
    io::stdin().read_to_end(&mut bytes)?;
    let input: Input = serde_json::from_slice(&bytes)?;

    if input.schema_version != INPUT_SCHEMA {
        return Err(format!("unsupported input schema: {}", input.schema_version).into());
    }
    if input.formalism_profile != FORMALISM_PROFILE {
        return Err(format!(
            "unsupported formalism profile: {}",
            input.formalism_profile
        )
        .into());
    }
    if !(2..=MAX_NODES).contains(&input.n_nodes) {
        return Err(format!("n_nodes must be in 2..={MAX_NODES}").into());
    }
    if !validate_hex_sha256(&input.model_sha256) {
        return Err("model_sha256 must contain exactly 64 hexadecimal characters".into());
    }

    let expected_rows = 1usize << input.n_nodes;
    if input.tpm.len() != expected_rows {
        return Err(format!(
            "TPM must have {expected_rows} state rows, got {}",
            input.tpm.len()
        )
        .into());
    }
    for (row_index, row) in input.tpm.iter().enumerate() {
        if row.len() != input.n_nodes {
            return Err(format!(
                "TPM row {row_index} must have {} node probabilities, got {}",
                input.n_nodes,
                row.len()
            )
            .into());
        }
        for (node_index, probability) in row.iter().enumerate() {
            if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
                return Err(format!(
                    "TPM probability at row {row_index}, node {node_index} must be finite and in [0,1]"
                )
                .into());
            }
        }
    }
    if input.state.len() != input.n_nodes {
        return Err(format!(
            "state must have {} values, got {}",
            input.n_nodes,
            input.state.len()
        )
        .into());
    }
    if input.state.iter().any(|bit| *bit > 1) {
        return Err("state values must be binary integers (0 or 1)".into());
    }

    // Constructor assertions are safe only after dimensions and probabilities
    // have been checked above; malformed inputs must fail as errors, not panics.
    let tpm = TransitionProbabilityMatrix::new(input.tpm, input.n_nodes);
    let state: Vec<bool> = input.state.iter().map(|bit| *bit == 1).collect();
    let value = system_phi(&tpm, &state);
    if !value.is_finite() || value < 0.0 {
        return Err(format!("candidate score is invalid: {value}").into());
    }

    let output = json!({
        "schema_version": OUTPUT_SCHEMA,
        "metric_id": METRIC_ID,
        "formalism_profile": FORMALISM_PROFILE,
        "model_sha256": input.model_sha256,
        "n_nodes": input.n_nodes,
        "state": input.state,
        "value": value,
        "partition_search": "exhaustive_bipartitions_for_full_system_mechanism_only",
        "claim_ceiling": "candidate score; formal equivalence to PyPhi is not assumed"
    });
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("iit3_tpm_phi_adapter: {error}");
        std::process::exit(2);
    }
}
