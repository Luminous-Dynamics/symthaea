use std::fs;
use std::path::PathBuf;
use symthaea_sim_bridge::{EngineeringDomain, SimulationEvidence, SimulationRequest, SolverKind};
use symthaea_sim_digest::{canonical_output_sha256_v1, canonical_request_sha256_v1};
use symthaea_sim_worker::{
    SUPERVISOR_PROFILE_V1, SupervisedWorker, SupervisorError, SupervisorLimits,
    WORKER_PROTOCOL_V1,
};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples/extensions/hello-simulation")
}

fn fixture_bytes() -> (Vec<u8>, Vec<u8>) {
    let fixture = fixture_dir();
    let manifest = fs::read(fixture.join("manifest.json")).expect("read fixture manifest");
    let component = fs::read(
        fixture.join("target/wasm32-wasip2/release/symthaea_hello_simulation_extension.wasm"),
    )
    .expect("read release-built hello-simulation Component");
    (manifest, component)
}

fn request(id: &str) -> SimulationRequest {
    SimulationRequest::new(
        id,
        EngineeringDomain::Systems,
        SolverKind::Custom,
        "exercise supervised simulation worker",
    )
    .with_parameter("alpha", 2.5, "1", "supervisor integration fixture")
    .with_parameter("beta", 1.5, "1", "supervisor integration fixture")
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires release-built examples/extensions/hello-simulation Component"]
fn real_component_executes_out_of_process_with_parent_reverification() {
    let (manifest, component) = fixture_bytes();
    let request = request("supervised-worker-success");
    let expected_request = canonical_request_sha256_v1(&request).unwrap();

    let worker = SupervisedWorker::new(env!("CARGO_BIN_EXE_symthaea-sim-worker"))
        .with_limits(SupervisorLimits {
            // Wasmtime may reserve a large virtual address range for wasm32
            // memories. This is a conservative first supervisor bound, not a
            // claim of tight physical-memory containment.
            address_space_bytes: 16 * 1024 * 1024 * 1024,
            wall_time_ms: 20_000,
            ..SupervisorLimits::default()
        })
        .unwrap();

    let invocation = worker.execute(&manifest, &component, &request).unwrap();
    assert_eq!(invocation.supervisor_profile(), SUPERVISOR_PROFILE_V1);
    assert_ne!(invocation.worker_sha256(), [0; 32]);
    assert_eq!(invocation.worker().profile, WORKER_PROTOCOL_V1);
    assert_eq!(invocation.worker().request_sha256, hex(expected_request));
    assert_eq!(invocation.worker().extension_id, "org.example.hello-simulation");
    assert_eq!(invocation.worker().extension_version, "0.1.0");

    let result = invocation.result();
    assert_eq!(result.evidence, SimulationEvidence::default());
    assert_eq!(result.request_id, request.id);
    assert_eq!(result.metrics.len(), 1);
    assert_eq!(result.metrics[0].name, "fixture.parameter-sum");
    assert_eq!(result.metrics[0].value, 4.0);
    assert_eq!(
        invocation.worker().output_sha256,
        hex(canonical_output_sha256_v1(result).unwrap())
    );
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires release-built examples/extensions/hello-simulation manifest"]
fn non_reading_child_cannot_bypass_parent_wall_clock_deadline() {
    let (manifest, _) = fixture_bytes();
    let request = request("supervised-worker-timeout");
    let worker = SupervisedWorker::new(env!("CARGO_BIN_EXE_symthaea-sim-stall-worker"))
        .with_limits(SupervisorLimits {
            wall_time_ms: 100,
            ..SupervisorLimits::default()
        })
        .unwrap();

    let error = worker
        .execute(&manifest, b"small-stall-fixture", &request)
        .unwrap_err();
    assert!(matches!(error, SupervisorError::WallTimeExceeded(100)));
}

fn hex(bytes: [u8; 32]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in bytes {
        out.push(TABLE[(byte >> 4) as usize] as char);
        out.push(TABLE[(byte & 0x0f) as usize] as char);
    }
    out
}
