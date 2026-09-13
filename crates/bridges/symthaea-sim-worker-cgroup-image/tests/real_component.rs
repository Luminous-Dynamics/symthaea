// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::fs;
use std::path::PathBuf;
use symthaea_sim_bridge::{EngineeringDomain, SimulationEvidence, SimulationRequest, SolverKind};
use symthaea_sim_worker::{CGROUP_PLACEMENT_PROFILE_V1, SupervisorLimits, WorkerFrameLimits};
use symthaea_sim_worker_cgroup::{CgroupV2Lease, CgroupV2Limits, CGROUP_V2_PROFILE_V1};
use symthaea_sim_worker_cgroup_image::{
    execute_sealed_image_in_cgroup, SEALED_CGROUP_WORKER_PROFILE_V1,
};
use symthaea_sim_worker_image::{SealedWorkerImage, SEALED_WORKER_IMAGE_PROFILE_V1};

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
        "sealed-cgroup-real",
        EngineeringDomain::Systems,
        SolverKind::Custom,
        "execute exact sealed Component worker in exact cgroup",
    )
    .with_parameter("alpha", 2.5, "1", "sealed cgroup integration")
    .with_parameter("beta", 1.5, "1", "sealed cgroup integration")
}

fn lease() -> CgroupV2Lease {
    let root = std::env::var("SYMTHAEA_CGROUP_V2_DELEGATED_ROOT")
        .expect("workflow provides explicit cgroup-v2 resource root");
    CgroupV2Lease::create(
        root,
        &format!("symthaea-sealed-cgroup-{}", std::process::id()),
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
#[ignore = "requires explicit cgroup-v2 resource root, filesystem-contained worker, and release Component"]
fn replaced_source_path_cannot_change_sealed_cgroup_execution() {
    let (manifest, component) = fixture_bytes();
    let worker = PathBuf::from(
        std::env::var("SYMTHAEA_CGROUP_WORKER_BIN")
            .expect("workflow provides exact contained worker binary"),
    );
    let temp = std::env::temp_dir().join(format!(
        "symthaea-sealed-cgroup-worker-{}",
        std::process::id()
    ));
    fs::copy(&worker, &temp).unwrap();
    let image = SealedWorkerImage::from_path(&temp).unwrap();
    let image_sha = image.image_sha256();

    // Replace the source pathname after sealing. Execution must remain bound to
    // the immutable memfd bytes, not the pathname that originally supplied them.
    fs::write(&temp, b"replaced after sealing").unwrap();

    let lease = lease();
    let invocation = execute_sealed_image_in_cgroup(
        &image,
        &lease,
        &manifest,
        &component,
        &request(),
        SupervisorLimits::default(),
        WorkerFrameLimits::default(),
    )
    .unwrap();
    invocation.verify().unwrap();
    let evidence = invocation.evidence();
    evidence.verify().unwrap();

    assert_eq!(evidence.profile, SEALED_CGROUP_WORKER_PROFILE_V1);
    assert_eq!(evidence.image.profile, SEALED_WORKER_IMAGE_PROFILE_V1);
    assert_eq!(invocation.invocation().worker_sha256(), image_sha);
    assert_eq!(evidence.placement.profile, CGROUP_PLACEMENT_PROFILE_V1);
    assert_eq!(evidence.placement.cgroup.profile, CGROUP_V2_PROFILE_V1);
    assert_eq!(evidence.placement.cgroup.memory_max_bytes, 1024 * 1024 * 1024);
    assert_eq!(evidence.placement.cgroup.pids_max, 64);
    assert!(evidence.placement.cgroup.memory_oom_group);

    let result = invocation.invocation().result();
    assert_eq!(result.evidence, SimulationEvidence::default());
    assert_eq!(result.metrics.len(), 1);
    assert_eq!(result.metrics[0].name, "fixture.parameter-sum");
    assert_eq!(result.metrics[0].value, 4.0);
    assert_eq!(result.warnings.len(), 1);
    assert!(result.warnings[0].contains("not engineering evidence"));

    assert!(!lease.populated().unwrap());
    lease.remove_empty().unwrap();
    let _ = fs::remove_file(temp);
}
