use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use symthaea_extension_admission::{
    AdmissionContext, AdmissionCurrentnessSource, AdmissionRecord, AdmissionSubject, PrincipalId,
    Sha256Digest, TrustLevel,
};
use symthaea_extension_core::{CapabilityId, ExtensionManifest, PermissionSet};
use symthaea_sim_bridge::{EngineeringDomain, SimulationEvidence, SimulationRequest, SolverKind};
use symthaea_sim_deployment::{BoundSimulationDeployment, SimulationDeploymentError};
use symthaea_sim_worker::{SupervisorLimits, WorkerFrameLimits};
use symthaea_sim_worker_image::SealedWorkerImage;
use symthaea_sim_worker_qualification::{
    WorkerQualificationContext, WorkerQualificationCurrentnessSource, WorkerQualificationRecord,
};

#[derive(Debug)]
struct FixedAdmissionCurrentness(AdmissionContext);

impl AdmissionCurrentnessSource for FixedAdmissionCurrentness {
    fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
        Some(self.0)
    }
}

#[derive(Debug)]
struct FixedWorkerCurrentness(WorkerQualificationContext);

impl WorkerQualificationCurrentnessSource for FixedWorkerCurrentness {
    fn current_context(&self, _worker_sha256: [u8; 32]) -> Option<WorkerQualificationContext> {
        Some(self.0)
    }
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples/extensions/hello-simulation")
}

fn exact_fixture_manifest() -> (Vec<u8>, ExtensionManifest) {
    let bytes = fs::read(fixture_dir().join("manifest.json")).expect("read fixture manifest");
    let manifest = serde_json::from_slice(&bytes).expect("parse fixture manifest");
    (bytes, manifest)
}

fn admission_for(
    manifest: &ExtensionManifest,
    manifest_bytes: &[u8],
    component_bytes: &[u8],
    generation: u64,
    trust_generation: u64,
) -> AdmissionRecord {
    AdmissionRecord::issue(
        manifest.id.clone(),
        manifest.version.clone(),
        Sha256Digest::new(Sha256::digest(manifest_bytes).into()),
        Sha256Digest::new(Sha256::digest(component_bytes).into()),
        Sha256Digest::new([0x55; 32]),
        PrincipalId::new("test.deployment-authority").unwrap(),
        None,
        TrustLevel::Community,
        vec![CapabilityId::new("engineering.simulation.custom")],
        PermissionSet::default(),
        generation,
        trust_generation,
    )
    .expect("issue exact admission")
}

fn request() -> SimulationRequest {
    SimulationRequest::new(
        "deployment-binding-real-component",
        EngineeringDomain::Systems,
        SolverKind::Custom,
        "execute exact admitted package through exact qualified worker",
    )
    .with_parameter("alpha", 2.5, "1", "deployment integration")
    .with_parameter("beta", 1.5, "1", "deployment integration")
}

fn decode_hex_32(value: &str) -> [u8; 32] {
    assert_eq!(value.len(), 64, "qualification digest must be 64 hex chars");
    let mut bytes = [0u8; 32];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let text = std::str::from_utf8(chunk).expect("hex is utf8");
        bytes[index] = u8::from_str_radix(text, 16).expect("qualification digest is hex");
    }
    bytes
}

#[test]
fn deployment_binds_wasm_package_without_rewriting_manifest_runtime() {
    let (manifest_bytes, manifest) = exact_fixture_manifest();
    let component = b"synthetic-component-bytes".to_vec();
    let admission_currentness = FixedAdmissionCurrentness(AdmissionContext::active(3, 4));
    let record = admission_for(&manifest, &manifest_bytes, &component, 3, 4);
    let admission = record
        .activate(&manifest, &admission_currentness)
        .expect("activate admission");

    let image = SealedWorkerImage::from_path(std::env::current_exe().unwrap())
        .expect("seal current test executable");
    let qualification_record =
        WorkerQualificationRecord::issue(image.image_sha256(), [0xa7; 32], 9).unwrap();
    let worker_currentness = FixedWorkerCurrentness(WorkerQualificationContext::active(9));
    let qualification = qualification_record
        .activate(&image, &worker_currentness)
        .expect("activate exact worker qualification");

    let deployment = BoundSimulationDeployment::issue(
        &admission,
        &admission_currentness,
        &qualification,
        &worker_currentness,
        manifest_bytes,
        component,
        image,
        SupervisorLimits::default(),
        WorkerFrameLimits::default(),
    )
    .expect("issue deployment binding");

    let evidence = deployment.evidence();
    assert_eq!(
        evidence.package_runtime,
        symthaea_extension_core::RuntimeKind::Wasm
    );
    assert_eq!(evidence.selected_extension, "org.example.hello-simulation");
    assert_eq!(evidence.worker_qualification_generation, 9);
    assert!(!evidence.deployment_sha256.chars().all(|ch| ch == '0'));
    deployment
        .verify(&admission, &qualification)
        .expect("reverify exact binding");
}

#[test]
fn deployment_rejects_payload_substitution_before_worker_use() {
    let (manifest_bytes, manifest) = exact_fixture_manifest();
    let admitted_component = b"admitted-component".to_vec();
    let substituted_component = b"substituted-component".to_vec();
    let admission_currentness = FixedAdmissionCurrentness(AdmissionContext::active(3, 4));
    let record = admission_for(
        &manifest,
        &manifest_bytes,
        &admitted_component,
        3,
        4,
    );
    let admission = record.activate(&manifest, &admission_currentness).unwrap();

    let image = SealedWorkerImage::from_path(std::env::current_exe().unwrap()).unwrap();
    let qualification_record =
        WorkerQualificationRecord::issue(image.image_sha256(), [0xa7; 32], 9).unwrap();
    let worker_currentness = FixedWorkerCurrentness(WorkerQualificationContext::active(9));
    let qualification = qualification_record
        .activate(&image, &worker_currentness)
        .unwrap();

    let error = BoundSimulationDeployment::issue(
        &admission,
        &admission_currentness,
        &qualification,
        &worker_currentness,
        manifest_bytes,
        substituted_component,
        image,
        SupervisorLimits::default(),
        WorkerFrameLimits::default(),
    )
    .expect_err("substituted payload must fail");
    assert!(matches!(
        error,
        SimulationDeploymentError::AdmissionDigestMismatch { field: "payload" }
    ));
}

#[test]
fn deployment_rejects_revoked_worker_qualification_at_issuance() {
    let (manifest_bytes, manifest) = exact_fixture_manifest();
    let component = b"synthetic-component-bytes".to_vec();
    let admission_currentness = FixedAdmissionCurrentness(AdmissionContext::active(3, 4));
    let record = admission_for(&manifest, &manifest_bytes, &component, 3, 4);
    let admission = record.activate(&manifest, &admission_currentness).unwrap();

    let image = SealedWorkerImage::from_path(std::env::current_exe().unwrap()).unwrap();
    let active_source = FixedWorkerCurrentness(WorkerQualificationContext::active(9));
    let qualification_record =
        WorkerQualificationRecord::issue(image.image_sha256(), [0xa7; 32], 9).unwrap();
    let qualification = qualification_record.activate(&image, &active_source).unwrap();
    let revoked_source = FixedWorkerCurrentness(WorkerQualificationContext::revoked(9));

    let error = BoundSimulationDeployment::issue(
        &admission,
        &admission_currentness,
        &qualification,
        &revoked_source,
        manifest_bytes,
        component,
        image,
        SupervisorLimits::default(),
        WorkerFrameLimits::default(),
    )
    .expect_err("revoked exact worker qualification must fail");
    assert!(matches!(
        error,
        SimulationDeploymentError::IssuanceWorkerCurrentness(_)
    ));
}

#[test]
#[ignore = "requires exact filesystem-contained worker and release-built hello-simulation Component"]
fn exact_admission_and_exact_qualified_worker_execute_real_component() {
    let fixture = fixture_dir();
    let manifest_bytes = fs::read(fixture.join("manifest.json")).unwrap();
    let manifest: ExtensionManifest = serde_json::from_slice(&manifest_bytes).unwrap();
    let component = fs::read(
        fixture.join("target/wasm32-wasip2/release/symthaea_hello_simulation_extension.wasm"),
    )
    .unwrap();

    let admission_currentness = FixedAdmissionCurrentness(AdmissionContext::active(21, 34));
    let record = admission_for(&manifest, &manifest_bytes, &component, 21, 34);
    let admission = record.activate(&manifest, &admission_currentness).unwrap();

    let worker_path = std::env::var("SYMTHAEA_FILESYSTEM_WORKER_BIN")
        .expect("workflow must provide exact filesystem-contained worker path");
    let qualification_evidence = decode_hex_32(
        &std::env::var("SYMTHAEA_WORKER_QUALIFICATION_EVIDENCE_SHA256")
            .expect("workflow must provide exact qualification evidence root"),
    );
    assert_ne!(qualification_evidence, [0; 32]);

    let image = SealedWorkerImage::from_path(worker_path).expect("seal exact contained worker");
    let qualification_record =
        WorkerQualificationRecord::issue(image.image_sha256(), qualification_evidence, 55).unwrap();
    let worker_currentness = FixedWorkerCurrentness(WorkerQualificationContext::active(55));
    let qualification = qualification_record
        .activate(&image, &worker_currentness)
        .expect("activate exact qualified worker");

    let deployment = BoundSimulationDeployment::issue(
        &admission,
        &admission_currentness,
        &qualification,
        &worker_currentness,
        manifest_bytes,
        component,
        image,
        SupervisorLimits::default(),
        WorkerFrameLimits::default(),
    )
    .expect("issue exact deployment");

    let invocation = deployment
        .execute(
            &admission,
            &admission_currentness,
            &qualification,
            &worker_currentness,
            &request(),
        )
        .expect("execute exact bound deployment");

    assert_eq!(invocation.deployment_sha256(), deployment.deployment_sha256());
    assert_eq!(
        invocation.worker_qualification_evidence_sha256(),
        qualification_evidence
    );
    let result = invocation.invocation().invocation().result();
    assert_eq!(result.evidence, SimulationEvidence::default());
    assert_eq!(result.metrics.len(), 1);
    assert_eq!(result.metrics[0].name, "fixture.parameter-sum");
    assert_eq!(result.metrics[0].value, 4.0);
}
