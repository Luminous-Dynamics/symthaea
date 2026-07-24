// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reference implementation of the external sparse-QP wire protocol.
//!
//! This worker deliberately uses Symthaea's deterministic active-set solver.
//! It exists to qualify transport, validation, and deployment wiring before a
//! production sparse solver is substituted. It processes one request and exits.

use std::io::{self, Read};

use symthaea_humanoid::{
    DenseEqualityQpSolver, EqualityQpSolverConfig, SparseQpWireRequest, SparseQpWireResponse,
};

const DEFAULT_BACKEND_ID: &str = "symthaea-reference-process-active-set-v1";

fn main() {
    let backend_id = std::env::var("SYMTHAEA_SPARSE_QP_BACKEND_ID")
        .unwrap_or_else(|_| DEFAULT_BACKEND_ID.to_string());
    let response = execute(&backend_id)
        .unwrap_or_else(|error| SparseQpWireResponse::failure(1, backend_id.clone(), error));
    match serde_json::to_string(&response) {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("failed to serialize sparse-QP response: {error}");
            std::process::exit(2);
        }
    }
    if response.error.is_some() {
        std::process::exit(1);
    }
}

fn execute(backend_id: &str) -> Result<SparseQpWireResponse, String> {
    if backend_id.trim().is_empty() {
        return Err("worker backend identity is empty".to_string());
    }
    let mut bytes = Vec::new();
    let mut input = io::stdin().take(8 * 1024 * 1024);
    input
        .read_to_end(&mut bytes)
        .map_err(|error| format!("failed to read sparse-QP request: {error}"))?;
    let request: SparseQpWireRequest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid sparse-QP request: {error}"))?;
    if !request.validate() {
        return Ok(SparseQpWireResponse::failure(
            request.request_id.max(1),
            backend_id,
            "request failed protocol validation",
        ));
    }
    if request.requested_backend_id != backend_id {
        return Ok(SparseQpWireResponse::failure(
            request.request_id,
            backend_id,
            format!(
                "requested backend {} does not match worker {}",
                request.requested_backend_id, backend_id
            ),
        ));
    }
    let solver = DenseEqualityQpSolver::with_config(EqualityQpSolverConfig::default());
    let solution = solver
        .solve_with_warm_start(&request.problem, request.warm_start.as_deref())
        .ok_or_else(|| "reference solver failed".to_string())?;
    Ok(SparseQpWireResponse::success(
        request.request_id,
        backend_id,
        request.warm_start.is_some(),
        solution,
    ))
}
