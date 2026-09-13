use std::fs;
use std::path::PathBuf;
use symthaea_sim_bridge::{EngineeringDomain, SimulationEvidence, SimulationRequest, SolverKind};
use symthaea_sim_worker_filesystem::WORKER_FILESYSTEM_PROFILE_V1;
use symthaea_sim_worker_image::SealedWorkerImage;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples/extensions/hello-simulation")
}

fn request() -> SimulationRequest {
    SimulationRequest::new(
        "filesystem-contained-worker-real-component",
        EngineeringDomain::Systems,
        SolverKind::Custom,
        "execute exact Component under Landlock filesystem confinement",
    )
    .with_parameter("alpha", 2.5, "1", "filesystem containment integration")
    .with_parameter("beta", 1.5, "1", "filesystem containment integration")
}

#[test]
#[ignore = "requires built filesystem-contained worker and release-built hello-simulation Component"]
fn landlock_contained_sealed_worker_executes_real_component() {
    let worker = PathBuf::from(
        std::env::var("SYMTHAEA_FILESYSTEM_WORKER_BIN")
            .expect("workflow must provide exact filesystem worker path"),
    );
    let image = SealedWorkerImage::from_path(&worker).expect("seal filesystem-contained worker");

    let fixture = fixture_dir();
    let manifest = fs::read(fixture.join("manifest.json")).expect("read exact manifest");
    let component = fs::read(
        fixture.join("target/wasm32-wasip2/release/symthaea_hello_simulation_extension.wasm"),
    )
    .expect("read exact release Component");

    let sealed = image
        .execute(&manifest, &component, &request())
        .expect("execute filesystem-contained sealed worker");
    sealed.verify().expect("verify sealed worker identity");

    assert_eq!(sealed.invocation().worker_sha256(), image.image_sha256());
    assert_eq!(
        sealed.invocation().result().evidence,
        SimulationEvidence::default()
    );
    assert_eq!(sealed.invocation().result().metrics.len(), 1);
    assert_eq!(
        sealed.invocation().result().metrics[0].name,
        "fixture.parameter-sum"
    );
    assert_eq!(sealed.invocation().result().metrics[0].value, 4.0);
    assert_eq!(
        WORKER_FILESYSTEM_PROFILE_V1,
        "symthaea.simulation.worker-filesystem.landlock-deny-content-v1"
    );
}
