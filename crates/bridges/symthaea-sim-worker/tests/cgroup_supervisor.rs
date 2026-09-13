// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::fs;
use std::path::PathBuf;
use symthaea_sim_bridge::{EngineeringDomain, SimulationEvidence, SimulationRequest, SolverKind};
use symthaea_sim_worker::{
    CGROUP_PLACEMENT_PROFILE_V1, SupervisedWorker, SupervisorError, SupervisorLimits,
};
use symthaea_sim_worker_cgroup::{CgroupV2Lease, CgroupV2Limits, CGROUP_V2_PROFILE_V1};

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
        "execute exact Component through pre-input cgroup placement",
    )
    .with_parameter("alpha", 2.5, "1", "cgroup supervisor integration")
    .with_parameter("beta", 1.5, "1", "cgroup supervisor integration")
}

fn lease(suffix: &str) -> CgroupV2Lease {
    let root = std::env::var("SYMTHAEA_CGROUP_V2_DELEGATED_ROOT")
        .expect("workflow provides explicit cgroup-v2 resource root");
    let leaf = format!("symthaea-supervisor-{}-{suffix}", std::process::id());
    CgroupV2Lease::create(
        root,
        &leaf,
        CgroupV2Limits {
            memory_max_bytes: 1024 * 1024 * 1024,
            pids_max: 64,
            cpu_quota_us: 100_000,
            cpu_period_us: 100_000,
        },
    )
    .unwrap()
}

#[test]
#[ignore = "requires explicit cgroup-v2 resource root, contained worker, and release Component"]
fn real_component_is_placed_before_input_and_returns_technical_result() {
    let (manifest, component) = fixture_bytes();
    let worker = std::env::var("SYMTHAEA_CGROUP_WORKER_BIN")
        .expect("workflow provides exact contained worker binary");
    let lease = lease("real");
    let invocation = SupervisedWorker::new(worker)
        .execute_in_cgroup(&manifest, &component, &request("cgroup-real"), &lease)
        .unwrap();

    let placement = invocation
        .cgroup_placement()
        .expect("cgroup execution must carry placement evidence");
    placement.verify().unwrap();
    assert_eq!(placement.profile, CGROUP_PLACEMENT_PROFILE_V1);
    assert_eq!(placement.cgroup.profile, CGROUP_V2_PROFILE_V1);
    assert!(placement.worker_pid > 0);
    assert_eq!(placement.cgroup.memory_max_bytes, 1024 * 1024 * 1024);
    assert_eq!(placement.cgroup.pids_max, 64);
    assert_eq!(placement.cgroup.cpu_quota_us, 100_000);
    assert_eq!(placement.cgroup.cpu_period_us, 100_000);
    assert!(placement.cgroup.memory_oom_group);

    let result = invocation.result();
    assert_eq!(result.evidence, SimulationEvidence::default());
    assert_eq!(result.metrics.len(), 1);
    assert_eq!(result.metrics[0].name, "fixture.parameter-sum");
    assert_eq!(result.metrics[0].value, 4.0);
    assert_eq!(result.warnings.len(), 1);
    assert!(result.warnings[0].contains("not engineering evidence"));

    assert!(!lease.populated().unwrap());
    lease.remove_empty().unwrap();
}

#[test]
#[ignore = "requires explicit cgroup-v2 resource root and stall worker binary"]
fn nonreading_worker_is_still_killed_by_parent_deadline_under_cgroup() {
    let (manifest, component) = fixture_bytes();
    let worker = std::env::var("SYMTHAEA_CGROUP_STALL_WORKER_BIN")
        .expect("workflow provides exact stall worker binary");
    let lease = lease("stall");
    let limits = SupervisorLimits {
        wall_time_ms: 100,
        ..SupervisorLimits::default()
    };
    let error = SupervisedWorker::new(worker)
        .with_limits(limits)
        .unwrap()
        .execute_in_cgroup(&manifest, &component, &request("cgroup-stall"), &lease)
        .expect_err("non-reading worker must hit parent wall-clock deadline");
    assert!(matches!(error, SupervisorError::WallTimeExceeded(100)));

    assert!(!lease.populated().unwrap());
    lease.remove_empty().unwrap();
}
