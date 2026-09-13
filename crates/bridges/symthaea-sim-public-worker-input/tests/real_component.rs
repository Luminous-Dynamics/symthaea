// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use symthaea_sim_bridge::{EngineeringDomain, SimulationEvidence, SimulationRequest, SolverKind};
use symthaea_sim_public_worker_input::{
    PUBLIC_MAX_COMPONENT_BYTES_V1, PUBLIC_MAX_MANIFEST_BYTES_V1,
    PUBLIC_MAX_REQUEST_JSON_BYTES_V1, PUBLIC_MAX_RESPONSE_JSON_BYTES_V1,
    PUBLIC_WORKER_INPUT_PROFILE_V1, RESPONSE_LIMIT_SEMANTICS_V1,
    execute_public_first_instruction_worker_with_rlimits,
};
use symthaea_sim_worker::SupervisorLimits;
use symthaea_sim_worker_cgroup::{CgroupV2Lease, CgroupV2Limits};
use symthaea_sim_worker_image::SealedWorkerImage;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples/extensions/hello-simulation")
}

fn fixture_bytes() -> (Vec<u8>, Vec<u8>) {
    let dir = fixture_dir();
    let manifest = fs::read(dir.join("manifest.json")).unwrap();
    let component = fs::read(
        dir.join("target/wasm32-wasip2/release/symthaea_hello_simulation_extension.wasm"),
    )
    .unwrap();
    (manifest, component)
}

fn request() -> SimulationRequest {
    SimulationRequest::new(
        "public-worker-input-real",
        EngineeringDomain::Systems,
        SolverKind::Custom,
        "execute exact public Component only after bounded pre-spawn input preparation",
    )
    .with_parameter("alpha", 2.5, "1", "public worker input integration")
    .with_parameter("beta", 1.5, "1", "public worker input integration")
}

fn lease() -> CgroupV2Lease {
    let root = std::env::var("SYMTHAEA_CGROUP_V2_DELEGATED_ROOT")
        .expect("workflow provides explicit cgroup-v2 resource root");
    let leaf = format!("symthaea-public-worker-input-{}", std::process::id());
    CgroupV2Lease::create(
        root,
        &leaf,
        CgroupV2Limits {
            memory_max_bytes: 2 * 1024 * 1024 * 1024,
            pids_max: 64,
            cpu_quota_us: 100_000,
            cpu_period_us: 100_000,
        },
    )
    .unwrap()
}

fn wait_empty_then_remove(lease: CgroupV2Lease) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while lease.populated().unwrap() {
        if Instant::now() >= deadline {
            let _ = lease.kill_all();
            panic!("public worker input cgroup remained populated");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    lease.remove_empty().unwrap();
}

fn real_limits() -> SupervisorLimits {
    SupervisorLimits {
        address_space_bytes: 16 * 1024 * 1024 * 1024,
        cpu_seconds: 15,
        file_size_bytes: 1024 * 1024,
        open_files: 32,
        process_count: 64,
        wall_time_ms: 20_000,
        max_stdout_bytes: 16 * 1024 * 1024,
        max_stderr_bytes: 256 * 1024,
    }
}

#[test]
#[ignore = "requires explicit cgroup root, release Component and filesystem-contained worker"]
fn real_component_executes_only_after_public_input_preflight() {
    let (manifest, component) = fixture_bytes();
    let request = request();
    let worker = std::env::var("SYMTHAEA_PUBLIC_WORKER_BIN")
        .expect("workflow provides exact filesystem-contained worker binary");
    let image = SealedWorkerImage::from_path(worker).unwrap();
    let lease = lease();

    let invocation = execute_public_first_instruction_worker_with_rlimits(
        &image,
        &lease,
        &manifest,
        &component,
        &request,
        real_limits(),
    )
    .unwrap();

    let evidence = invocation.evidence();
    evidence.verify().unwrap();
    assert_eq!(evidence.profile, PUBLIC_WORKER_INPUT_PROFILE_V1);
    assert_eq!(evidence.max_manifest_bytes, PUBLIC_MAX_MANIFEST_BYTES_V1);
    assert_eq!(evidence.max_component_bytes, PUBLIC_MAX_COMPONENT_BYTES_V1);
    assert_eq!(evidence.max_request_json_bytes, PUBLIC_MAX_REQUEST_JSON_BYTES_V1);
    assert_eq!(evidence.max_response_json_bytes, PUBLIC_MAX_RESPONSE_JSON_BYTES_V1);
    assert_eq!(evidence.observed_manifest_bytes, manifest.len() as u64);
    assert_eq!(evidence.observed_component_bytes, component.len() as u64);
    assert!(evidence.observed_request_json_bytes > 0);
    assert!(evidence.pre_spawn_raw_input_gate);
    assert!(evidence.bounded_request_json_counter);
    assert!(evidence.host_input_ceiling_parity);
    assert_eq!(evidence.response_limit_semantics, RESPONSE_LIMIT_SEMANTICS_V1);
    assert!(evidence.inner.legacy_preexec_rlimits_applied);
    assert!(evidence.inner.parent_wall_time_enforced);
    assert!(evidence.inner.parent_output_limits_enforced);
    assert!(!evidence.inner.containment_profiles_established_by_this_layer);

    let result = invocation.result();
    assert_eq!(result.evidence, SimulationEvidence::default());
    assert_eq!(result.metrics.len(), 1);
    assert_eq!(result.metrics[0].name, "fixture.parameter-sum");
    assert_eq!(result.metrics[0].value, 4.0);
    assert_eq!(result.warnings.len(), 1);
    assert!(result.warnings[0].contains("not engineering evidence"));

    assert!(!lease.populated().unwrap());
    wait_empty_then_remove(lease);
}
