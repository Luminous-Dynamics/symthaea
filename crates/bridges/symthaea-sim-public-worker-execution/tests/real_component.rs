// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use symthaea_sim_bridge::{EngineeringDomain, SimulationEvidence, SimulationRequest, SolverKind};
use symthaea_sim_public_worker_execution::{
    PUBLIC_MAX_RESPONSE_FRAME_BYTES_V1, PUBLIC_RESPONSE_FRAME_OVERHEAD_BYTES_V1,
    PUBLIC_WORKER_EXECUTION_PROFILE_V1, execute_public_worker_v1, public_cgroup_limits_v1,
    public_supervisor_limits_v1,
};
use symthaea_sim_worker_cgroup::CgroupV2Lease;
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
        "public-worker-execution-real",
        EngineeringDomain::Systems,
        SolverKind::Custom,
        "execute exact public Component under one frozen resource envelope",
    )
    .with_parameter("alpha", 2.5, "1", "public worker execution integration")
    .with_parameter("beta", 1.5, "1", "public worker execution integration")
}

fn lease() -> CgroupV2Lease {
    let root = std::env::var("SYMTHAEA_CGROUP_V2_DELEGATED_ROOT")
        .expect("workflow provides explicit cgroup-v2 resource root");
    let leaf = format!("symthaea-public-worker-execution-{}", std::process::id());
    CgroupV2Lease::create(root, &leaf, public_cgroup_limits_v1()).unwrap()
}

fn wait_empty_then_remove(lease: CgroupV2Lease) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while lease.populated().unwrap() {
        if Instant::now() >= deadline {
            let _ = lease.kill_all();
            panic!("public worker execution cgroup remained populated");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    lease.remove_empty().unwrap();
}

#[test]
#[ignore = "requires explicit cgroup root, release Component and filesystem-contained worker"]
fn real_component_executes_under_exact_public_resource_profile() {
    let (manifest, component) = fixture_bytes();
    let worker = std::env::var("SYMTHAEA_PUBLIC_EXECUTION_WORKER_BIN")
        .expect("workflow provides exact filesystem-contained worker binary");
    let image = SealedWorkerImage::from_path(worker).unwrap();
    let lease = lease();

    let invocation = execute_public_worker_v1(
        &image,
        &lease,
        &manifest,
        &component,
        &request(),
    )
    .unwrap();

    let evidence = invocation.evidence();
    evidence.verify().unwrap();
    assert_eq!(evidence.profile, PUBLIC_WORKER_EXECUTION_PROFILE_V1);
    assert!(evidence.exact_supervisor_profile);
    assert!(evidence.exact_cgroup_profile);
    assert!(evidence.framed_response_fits_stdout_exactly);
    assert_eq!(
        evidence.response_frame_overhead_bytes,
        PUBLIC_RESPONSE_FRAME_OVERHEAD_BYTES_V1
    );
    assert_eq!(
        evidence.max_response_frame_bytes,
        PUBLIC_MAX_RESPONSE_FRAME_BYTES_V1
    );
    assert_eq!(evidence.max_stdout_bytes, PUBLIC_MAX_RESPONSE_FRAME_BYTES_V1);
    assert_eq!(
        evidence.inner.inner.parent_max_stdout_bytes,
        PUBLIC_MAX_RESPONSE_FRAME_BYTES_V1
    );
    assert_eq!(
        evidence.inner.inner.launch.base.target.memory_max_bytes,
        public_cgroup_limits_v1().memory_max_bytes
    );
    assert_eq!(
        evidence.inner.inner.launch.base.target.pids_max,
        public_cgroup_limits_v1().pids_max
    );
    assert_eq!(
        evidence.inner.inner.launch.base.target.cpu_quota_us,
        public_cgroup_limits_v1().cpu_quota_us
    );
    assert_eq!(
        evidence.inner.inner.launch.base.target.cpu_period_us,
        public_cgroup_limits_v1().cpu_period_us
    );
    assert_eq!(
        evidence.inner.inner.launch.limits.address_space_bytes,
        public_supervisor_limits_v1().address_space_bytes
    );
    assert!(evidence.inner.inner.legacy_preexec_rlimits_applied);
    assert!(evidence.inner.inner.parent_wall_time_enforced);
    assert!(evidence.inner.inner.parent_output_limits_enforced);
    assert!(!evidence.inner.inner.containment_profiles_established_by_this_layer);

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
