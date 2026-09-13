// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use symthaea_sim_bridge::{EngineeringDomain, SimulationEvidence, SimulationRequest, SolverKind};
use symthaea_sim_first_instruction_worker::{
    execute_first_instruction_worker, FirstInstructionWorkerError, FirstInstructionWorkerLimits,
    FIRST_INSTRUCTION_WORKER_PROFILE_V1,
};
use symthaea_sim_worker::WorkerFrameLimits;
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

fn request(id: &str) -> SimulationRequest {
    SimulationRequest::new(
        id,
        EngineeringDomain::Systems,
        SolverKind::Custom,
        "execute exact Component from worker born directly in cgroup",
    )
    .with_parameter("alpha", 2.5, "1", "first-instruction integration")
    .with_parameter("beta", 1.5, "1", "first-instruction integration")
}

fn lease(suffix: &str) -> CgroupV2Lease {
    let root = std::env::var("SYMTHAEA_CGROUP_V2_DELEGATED_ROOT")
        .expect("workflow provides explicit cgroup-v2 resource root");
    let leaf = format!("symthaea-first-instruction-{}-{suffix}", std::process::id());
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
            panic!("first-instruction worker cgroup remained populated");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    lease.remove_empty().unwrap();
}

#[test]
#[ignore = "requires explicit cgroup root, release Component and filesystem-contained worker"]
fn real_component_executes_from_worker_born_in_resource_domain() {
    let (manifest, component) = fixture_bytes();
    let worker = std::env::var("SYMTHAEA_FIRST_INSTRUCTION_WORKER_BIN")
        .expect("workflow provides exact filesystem-contained worker binary");
    let image = SealedWorkerImage::from_path(worker).unwrap();
    let lease = lease("real");

    let invocation = execute_first_instruction_worker(
        &image,
        &lease,
        &manifest,
        &component,
        &request("first-instruction-real"),
        WorkerFrameLimits::default(),
        FirstInstructionWorkerLimits::default(),
    )
    .unwrap();

    invocation.evidence().verify().unwrap();
    assert_eq!(invocation.evidence().profile, FIRST_INSTRUCTION_WORKER_PROFILE_V1);
    assert!(invocation.evidence().worker_born_in_target_cgroup);
    assert!(invocation.evidence().worker_exec_in_target_cgroup);
    assert!(
        invocation
            .evidence()
            .worker_runtime_allocations_begin_after_exec_in_target_cgroup
    );
    assert!(!invocation.evidence().inherited_parent_memory_recharged);
    assert!(!invocation.evidence().legacy_preexec_rlimits_applied);
    assert!(
        !invocation
            .evidence()
            .containment_profiles_established_by_this_layer
    );

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

#[test]
#[ignore = "requires explicit cgroup root and exact non-reading stall worker"]
fn nonreading_worker_is_killed_by_parent_deadline_without_orphaning_cgroup() {
    let (manifest, component) = fixture_bytes();
    let worker = std::env::var("SYMTHAEA_FIRST_INSTRUCTION_STALL_BIN")
        .expect("workflow provides exact stall worker binary");
    let image = SealedWorkerImage::from_path(worker).unwrap();
    let lease = lease("stall");
    let limits = FirstInstructionWorkerLimits {
        wall_time_ms: 100,
        ..FirstInstructionWorkerLimits::default()
    };

    let error = execute_first_instruction_worker(
        &image,
        &lease,
        &manifest,
        &component,
        &request("first-instruction-stall"),
        WorkerFrameLimits::default(),
        limits,
    )
    .expect_err("non-reading worker must hit parent wall deadline");
    assert!(matches!(error, FirstInstructionWorkerError::WallTimeExceeded(100)));

    assert!(!lease.populated().unwrap());
    wait_empty_then_remove(lease);
}
