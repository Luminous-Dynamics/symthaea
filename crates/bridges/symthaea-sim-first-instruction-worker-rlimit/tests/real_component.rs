// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use symthaea_sim_bridge::{EngineeringDomain, SimulationEvidence, SimulationRequest, SolverKind};
use symthaea_sim_first_instruction_worker_rlimit::{
    FIRST_INSTRUCTION_WORKER_RLIMIT_PROFILE_V1, FirstInstructionWorkerRlimitError,
    execute_first_instruction_worker_with_rlimits,
};
use symthaea_sim_worker::{SupervisorLimits, WorkerFrameLimits};
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
        "execute exact Component with first-instruction cgroup and pre-exec rlimits",
    )
    .with_parameter("alpha", 2.5, "1", "rlimit first-instruction integration")
    .with_parameter("beta", 1.5, "1", "rlimit first-instruction integration")
}

fn lease(suffix: &str) -> CgroupV2Lease {
    let root = std::env::var("SYMTHAEA_CGROUP_V2_DELEGATED_ROOT")
        .expect("workflow provides explicit cgroup-v2 resource root");
    let leaf = format!("symthaea-first-instruction-rlimit-{}-{suffix}", std::process::id());
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
            panic!("first-instruction rlimit worker cgroup remained populated");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    lease.remove_empty().unwrap();
}

fn real_limits() -> SupervisorLimits {
    SupervisorLimits {
        // Wasmtime can reserve a large virtual address range. RLIMIT_AS is not
        // physical-memory containment; cgroup memory.max remains separate.
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
fn real_component_executes_with_first_instruction_rlimits_and_parent_bounds() {
    let (manifest, component) = fixture_bytes();
    let worker = std::env::var("SYMTHAEA_FIRST_INSTRUCTION_RLIMIT_WORKER_BIN")
        .expect("workflow provides exact filesystem-contained worker binary");
    let image = SealedWorkerImage::from_path(worker).unwrap();
    let lease = lease("real");
    let limits = real_limits();

    let invocation = execute_first_instruction_worker_with_rlimits(
        &image,
        &lease,
        &manifest,
        &component,
        &request("first-instruction-rlimit-real"),
        WorkerFrameLimits::default(),
        limits,
    )
    .unwrap();

    let evidence = invocation.evidence();
    evidence.verify().unwrap();
    assert_eq!(evidence.profile, FIRST_INSTRUCTION_WORKER_RLIMIT_PROFILE_V1);
    assert!(evidence.worker_born_in_target_cgroup);
    assert!(evidence.worker_exec_in_target_cgroup);
    assert!(evidence.worker_runtime_allocations_begin_after_exec_in_target_cgroup);
    assert!(!evidence.inherited_parent_memory_recharged);
    assert!(evidence.legacy_preexec_rlimits_applied);
    assert!(evidence.parent_wall_time_enforced);
    assert!(evidence.parent_output_limits_enforced);
    assert!(!evidence.containment_profiles_established_by_this_layer);
    assert_eq!(evidence.launch.limits.address_space_bytes, limits.address_space_bytes);
    assert_eq!(evidence.launch.limits.cpu_seconds, limits.cpu_seconds);
    assert_eq!(evidence.launch.limits.file_size_bytes, limits.file_size_bytes);
    assert_eq!(evidence.launch.limits.open_files, limits.open_files);
    assert_eq!(evidence.launch.limits.process_count, limits.process_count);
    assert_eq!(evidence.parent_wall_time_ms, limits.wall_time_ms);
    assert_eq!(evidence.parent_max_stdout_bytes, limits.max_stdout_bytes);
    assert_eq!(evidence.parent_max_stderr_bytes, limits.max_stderr_bytes);

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
fn nonreading_worker_still_hits_parent_deadline_under_preexec_rlimits() {
    let (manifest, component) = fixture_bytes();
    let worker = std::env::var("SYMTHAEA_FIRST_INSTRUCTION_RLIMIT_STALL_BIN")
        .expect("workflow provides exact stall worker binary");
    let image = SealedWorkerImage::from_path(worker).unwrap();
    let lease = lease("stall");
    let limits = SupervisorLimits {
        wall_time_ms: 100,
        ..real_limits()
    };

    let error = execute_first_instruction_worker_with_rlimits(
        &image,
        &lease,
        &manifest,
        &component,
        &request("first-instruction-rlimit-stall"),
        WorkerFrameLimits::default(),
        limits,
    )
    .expect_err("non-reading worker must hit parent wall deadline");
    assert!(matches!(
        error,
        FirstInstructionWorkerRlimitError::WallTimeExceeded(100)
    ));

    assert!(!lease.populated().unwrap());
    wait_empty_then_remove(lease);
}
