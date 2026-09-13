use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use symthaea_extension_admission::{
    AdmissionContext, AdmissionCurrentnessSource, AdmissionRecord, AdmissionSubject, PrincipalId,
    Sha256Digest, TrustLevel,
};
use symthaea_extension_core::{CapabilityId, ExtensionManifest, PermissionSet};
use symthaea_sim_deployment::BoundSimulationDeployment;
use symthaea_sim_first_instruction_deployment::{
    BoundFirstInstructionDeployment, FirstInstructionResourcePolicy,
};
use symthaea_sim_first_instruction_worker::FirstInstructionWorkerLimits;
use symthaea_sim_worker::{SupervisorLimits, WorkerFrameLimits};
use symthaea_sim_worker_cgroup::CgroupV2Limits;
use symthaea_sim_worker_image::SealedWorkerImage;
use symthaea_sim_worker_qualification::{
    WorkerQualificationContext, WorkerQualificationCurrentnessSource, WorkerQualificationRecord,
};

#[derive(Debug)]
struct AdmissionCurrentness;

impl AdmissionCurrentnessSource for AdmissionCurrentness {
    fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
        Some(AdmissionContext::active(3, 4))
    }
}

#[derive(Debug)]
struct WorkerCurrentness;

impl WorkerQualificationCurrentnessSource for WorkerCurrentness {
    fn current_context(&self, _worker_sha256: [u8; 32]) -> Option<WorkerQualificationContext> {
        Some(WorkerQualificationContext::active(9))
    }
}

fn manifest_bytes() -> (Vec<u8>, ExtensionManifest) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples/extensions/hello-simulation/manifest.json");
    let bytes = fs::read(path).expect("read canonical simulation manifest");
    let manifest = serde_json::from_slice(&bytes).expect("parse canonical simulation manifest");
    (bytes, manifest)
}

#[test]
fn nested_deployment_evidence_is_independently_recomputed() {
    let (manifest_bytes, manifest) = manifest_bytes();
    let component = b"synthetic-component".to_vec();
    let admission_currentness = AdmissionCurrentness;
    let worker_currentness = WorkerCurrentness;
    let admission_record = AdmissionRecord::issue(
        manifest.id.clone(),
        manifest.version.clone(),
        Sha256Digest::new(Sha256::digest(&manifest_bytes).into()),
        Sha256Digest::new(Sha256::digest(&component).into()),
        Sha256Digest::new([0x55; 32]),
        PrincipalId::new("test.first-instruction-evidence").unwrap(),
        None,
        TrustLevel::Community,
        vec![CapabilityId::new("engineering.simulation.custom")],
        PermissionSet::default(),
        3,
        4,
    )
    .unwrap();
    let admission = admission_record
        .activate(&manifest, &admission_currentness)
        .unwrap();

    let qualification_image =
        SealedWorkerImage::from_path(std::env::current_exe().unwrap()).unwrap();
    let qualification_record =
        WorkerQualificationRecord::issue(qualification_image.image_sha256(), [0xa7; 32], 9)
            .unwrap();
    let qualification = qualification_record
        .activate(&qualification_image, &worker_currentness)
        .unwrap();

    let base = BoundSimulationDeployment::issue(
        &admission,
        &admission_currentness,
        &qualification,
        &worker_currentness,
        manifest_bytes.clone(),
        component.clone(),
        SealedWorkerImage::from_path(std::env::current_exe().unwrap()).unwrap(),
        SupervisorLimits::default(),
        WorkerFrameLimits::default(),
    )
    .unwrap();
    let base_evidence = base.evidence();
    let stronger = BoundFirstInstructionDeployment::issue(
        &base,
        &admission,
        &admission_currentness,
        &qualification,
        &worker_currentness,
        manifest_bytes,
        component,
        SealedWorkerImage::from_path(std::env::current_exe().unwrap()).unwrap(),
        FirstInstructionResourcePolicy {
            cgroup: CgroupV2Limits::default(),
            worker: FirstInstructionWorkerLimits {
                wall_time_ms: base_evidence.supervisor.wall_time_ms,
                max_stdout_bytes: base_evidence.supervisor.max_stdout_bytes,
                max_stderr_bytes: base_evidence.supervisor.max_stderr_bytes,
            },
        },
    )
    .unwrap();

    let evidence = stronger.evidence();
    evidence.verify().expect("untampered evidence verifies");

    let mut transport = evidence.clone();
    transport.base_deployment.transport.push_str(".tampered");
    assert!(transport.verify().is_err());

    let mut worker_protocol = evidence.clone();
    worker_protocol
        .base_deployment
        .worker_protocol
        .push_str(".tampered");
    assert!(worker_protocol.verify().is_err());

    let mut supervisor = evidence.clone();
    supervisor.base_deployment.supervisor.wall_time_ms += 1;
    assert!(supervisor.verify().is_err());

    let mut frame = evidence.clone();
    frame.base_deployment.frame.max_response_json_bytes += 1;
    assert!(frame.verify().is_err());

    let mut selected = evidence;
    selected.base_deployment.selected_version.push_str("-tampered");
    assert!(selected.verify().is_err());
}
