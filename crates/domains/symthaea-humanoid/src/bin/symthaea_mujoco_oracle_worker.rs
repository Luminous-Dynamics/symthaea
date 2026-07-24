// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! One-shot independently buildable MuJoCo dynamics-oracle worker.

use std::io::Read;

use symthaea_humanoid::{
    HumanoidPhysicsSimulator, MuJoCoHumanoidSimulator, MujocoOracleWorkerRequest,
    MujocoOracleWorkerResponse,
};

const COMPILED_GENERATOR_ID: &str = match option_env!("SYMTHAEA_ORACLE_GENERATOR_ID") {
    Some(value) => value,
    None => "symthaea-mujoco-oracle-worker",
};
const COMPILED_GENERATOR_BUILD_ID: &str = match option_env!("SYMTHAEA_ORACLE_GENERATOR_BUILD_ID") {
    Some(value) => value,
    None => env!("CARGO_PKG_VERSION"),
};
const COMPILED_ENGINE_ID: &str = match option_env!("SYMTHAEA_ORACLE_ENGINE_ID") {
    Some(value) => value,
    None => "mujoco-rs-4.0.1",
};

fn main() {
    let mut payload = Vec::new();
    if std::io::stdin()
        .take(64 * 1024 * 1024 + 1)
        .read_to_end(&mut payload)
        .is_err()
        || payload.is_empty()
        || payload.len() > 64 * 1024 * 1024
    {
        eprintln!("invalid or oversized MuJoCo oracle request");
        std::process::exit(2);
    }
    let request: MujocoOracleWorkerRequest = match serde_json::from_slice(&payload) {
        Ok(request) if request.validate() => request,
        _ => {
            eprintln!("malformed MuJoCo oracle request");
            std::process::exit(2);
        }
    };
    if request.generator_id != COMPILED_GENERATOR_ID
        || request.generator_build_id != COMPILED_GENERATOR_BUILD_ID
        || request.engine_id != COMPILED_ENGINE_ID
        || request.candidate_build_id == COMPILED_GENERATOR_BUILD_ID
    {
        eprintln!("request identity does not match this compiled oracle worker");
        std::process::exit(2);
    }
    let response = generate(&request)
        .unwrap_or_else(|error| MujocoOracleWorkerResponse::failure(&request, error.to_string()));
    match serde_json::to_writer(std::io::stdout(), &response) {
        Ok(()) if response.error.is_none() => {}
        Ok(()) => std::process::exit(1),
        Err(error) => {
            eprintln!("failed to encode MuJoCo oracle response: {error}");
            std::process::exit(2);
        }
    }
}

fn generate(request: &MujocoOracleWorkerRequest) -> anyhow::Result<MujocoOracleWorkerResponse> {
    let mut simulator = MuJoCoHumanoidSimulator::for_morphology(request.morphology)?;
    simulator
        .set_generalized_state(&request.generalized_position, &request.generalized_velocity)?;
    let oracle = simulator
        .floating_base_dynamics_snapshot()
        .ok_or_else(|| anyhow::anyhow!("MuJoCo did not expose floating-base dynamics"))?;
    Ok(MujocoOracleWorkerResponse::success(request, oracle))
}
