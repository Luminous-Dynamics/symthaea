// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use sha2::{Digest, Sha256};
use std::io::{self, BufReader, BufWriter};
use symthaea_extension_simulation_host::SimulationComponentHost;
use symthaea_sim_digest::{canonical_output_sha256_v1, canonical_request_sha256_v1};
use symthaea_sim_worker::{
    WORKER_PROTOCOL_V1, WorkerFailure, WorkerFailureKind, WorkerFrameLimits, WorkerResponse,
    WorkerSuccess, read_request_frame, write_response_frame,
};

fn main() {
    let limits = WorkerFrameLimits::default();
    let response = match run(limits) {
        Ok(success) => WorkerResponse::Ok(success),
        Err(failure) => WorkerResponse::Err(failure),
    };

    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());
    if let Err(error) = write_response_frame(&mut writer, &response, limits) {
        eprintln!("failed to encode worker response: {error}");
        std::process::exit(70);
    }
}

fn run(limits: WorkerFrameLimits) -> Result<WorkerSuccess, WorkerFailure> {
    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let frame = read_request_frame(&mut reader, limits).map_err(|error| WorkerFailure {
        kind: WorkerFailureKind::InvalidFrame,
        message: error.to_string(),
    })?;

    frame.request.validate().map_err(|error| WorkerFailure {
        kind: WorkerFailureKind::InvalidRequest,
        message: error.to_string(),
    })?;

    let request_sha256 = canonical_request_sha256_v1(&frame.request).map_err(|error| WorkerFailure {
        kind: WorkerFailureKind::CanonicalizationFailure,
        message: error.to_string(),
    })?;
    let manifest_sha256: [u8; 32] = Sha256::digest(&frame.manifest_bytes).into();
    let component_sha256: [u8; 32] = Sha256::digest(&frame.component_bytes).into();

    let invocation = SimulationComponentHost::default()
        .invoke(
            &frame.manifest_bytes,
            &frame.component_bytes,
            &frame.request,
        )
        .map_err(|error| WorkerFailure {
            kind: WorkerFailureKind::HostFailure,
            message: error.to_string(),
        })?;

    if invocation.manifest_sha256() != manifest_sha256 {
        return Err(WorkerFailure {
            kind: WorkerFailureKind::Internal,
            message: "runtime manifest digest disagrees with worker preflight".into(),
        });
    }
    if invocation.component_sha256() != component_sha256 {
        return Err(WorkerFailure {
            kind: WorkerFailureKind::Internal,
            message: "runtime component digest disagrees with worker preflight".into(),
        });
    }

    let output_sha256 = canonical_output_sha256_v1(invocation.result()).map_err(|error| {
        WorkerFailure {
            kind: WorkerFailureKind::CanonicalizationFailure,
            message: error.to_string(),
        }
    })?;

    Ok(WorkerSuccess {
        profile: WORKER_PROTOCOL_V1.into(),
        manifest_sha256: hex_digest(manifest_sha256),
        component_sha256: hex_digest(component_sha256),
        request_sha256: hex_digest(request_sha256),
        output_sha256: hex_digest(output_sha256),
        extension_id: invocation.extension_id().to_owned(),
        extension_version: invocation.extension_version().to_owned(),
        control_wasm_profile: invocation.control_wasm_profile().into(),
        simulation_wasm_profile: invocation.simulation_wasm_profile().into(),
        wit_version: invocation.wit_version().into(),
        adapter_version: invocation.adapter_version().into(),
        result: invocation.into_result(),
    })
}

fn hex_digest(bytes: [u8; 32]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in bytes {
        out.push(TABLE[(byte >> 4) as usize] as char);
        out.push(TABLE[(byte & 0x0f) as usize] as char);
    }
    out
}
