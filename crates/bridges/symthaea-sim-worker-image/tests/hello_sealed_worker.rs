use std::fs;
use std::path::PathBuf;
use std::process;
use symthaea_sim_bridge::{EngineeringDomain, SimulationEvidence, SimulationRequest, SolverKind};
use symthaea_sim_worker_image::{SEALED_WORKER_IMAGE_PROFILE_V1, SealedWorkerImage};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples/extensions/hello-simulation")
}

fn request() -> SimulationRequest {
    SimulationRequest::new(
        "sealed-worker-real-component",
        EngineeringDomain::Systems,
        SolverKind::Custom,
        "execute exact Component through sealed worker image",
    )
    .with_parameter("alpha", 2.5, "1", "sealed worker integration")
    .with_parameter("beta", 1.5, "1", "sealed worker integration")
}

#[test]
#[ignore = "requires built worker binary and release-built hello-simulation Component"]
fn replaced_source_path_cannot_change_sealed_worker_execution() {
    let worker = PathBuf::from(
        std::env::var("SYMTHAEA_SIM_WORKER_BIN")
            .expect("workflow must provide exact built worker path"),
    );
    let temp_source = std::env::temp_dir().join(format!(
        "symthaea-worker-source-{}-{}",
        process::id(),
        std::thread::current().name().unwrap_or("integration")
    ));
    fs::copy(&worker, &temp_source).expect("copy exact worker source");

    let image = SealedWorkerImage::from_path(&temp_source).expect("seal worker image");
    assert_eq!(image.source_sha256(), image.image_sha256());
    assert_ne!(image.seal_mask(), 0);

    // Replace the ordinary filesystem pathname with unrelated bytes. Execution
    // must still resolve through the already-open immutable memfd descriptor.
    fs::remove_file(&temp_source).expect("unlink source after sealing");
    fs::write(&temp_source, b"hostile replacement after sealing")
        .expect("replace source pathname with different bytes");
    assert!(temp_source.exists());

    let fixture = fixture_dir();
    let manifest = fs::read(fixture.join("manifest.json")).expect("read exact manifest");
    let component = fs::read(
        fixture.join("target/wasm32-wasip2/release/symthaea_hello_simulation_extension.wasm"),
    )
    .expect("read exact release Component");

    let sealed = image
        .execute(&manifest, &component, &request())
        .expect("execute sealed worker image");
    sealed.verify().expect("verify sealed image binding");

    let evidence = sealed.evidence();
    assert_eq!(evidence.profile, SEALED_WORKER_IMAGE_PROFILE_V1);
    assert_eq!(evidence.source_sha256, evidence.image_sha256);
    assert_eq!(evidence.image_sha256, evidence.worker_sha256);
    assert!(!evidence.binding_sha256.chars().all(|ch| ch == '0'));

    let invocation = sealed.invocation();
    assert_eq!(invocation.worker_sha256(), image.image_sha256());
    assert_eq!(invocation.result().evidence, SimulationEvidence::default());
    assert_eq!(invocation.result().metrics.len(), 1);
    assert_eq!(invocation.result().metrics[0].name, "fixture.parameter-sum");
    assert_eq!(invocation.result().metrics[0].value, 4.0);

    let _ = fs::remove_file(temp_source);
}
