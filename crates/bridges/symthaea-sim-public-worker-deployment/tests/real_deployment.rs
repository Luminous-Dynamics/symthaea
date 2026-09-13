use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use symthaea_extension_admission::{
    AdmissionContext, AdmissionCurrentnessSource, AdmissionRecord, AdmissionSubject, PrincipalId,
    Sha256Digest, TrustLevel,
};
use symthaea_extension_core::{CapabilityId, ExtensionManifest, PermissionSet};
use symthaea_sim_bridge::{EngineeringDomain, SimulationEvidence, SimulationRequest, SolverKind};
use symthaea_sim_deployment::BoundSimulationDeployment;
use symthaea_sim_public_worker_deployment::{
    BoundPublicWorkerDeployment, PublicWorkerDeploymentError, PUBLIC_WORKER_DEPLOYMENT_PROFILE_V1,
};
use symthaea_sim_public_worker_execution::{
    public_cgroup_limits_v1, public_supervisor_limits_v1, PUBLIC_WORKER_EXECUTION_PROFILE_V1,
};
use symthaea_sim_public_worker_input::public_worker_frame_limits_v1;
use symthaea_sim_worker_cgroup::CgroupV2Lease;
use symthaea_sim_worker_image::SealedWorkerImage;
use symthaea_sim_worker_qualification::{
    WorkerQualificationContext, WorkerQualificationCurrentnessSource, WorkerQualificationRecord,
};

#[derive(Debug)]
struct SwitchAdmissionCurrentness {
    generation: u64,
    trust_generation: u64,
    revoke_on_call: AtomicUsize,
    calls: AtomicUsize,
}
impl SwitchAdmissionCurrentness {
    fn active(generation: u64, trust_generation: u64) -> Self {
        Self {
            generation,
            trust_generation,
            revoke_on_call: AtomicUsize::new(usize::MAX),
            calls: AtomicUsize::new(0),
        }
    }
    fn reset_revoke_on(&self, call: usize) {
        self.calls.store(0, Ordering::SeqCst);
        self.revoke_on_call.store(call, Ordering::SeqCst);
    }
}
impl AdmissionCurrentnessSource for SwitchAdmissionCurrentness {
    fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if call >= self.revoke_on_call.load(Ordering::SeqCst) {
            Some(AdmissionContext::revoked(self.generation, self.trust_generation))
        } else {
            Some(AdmissionContext::active(self.generation, self.trust_generation))
        }
    }
}

#[derive(Debug)]
struct SwitchWorkerCurrentness {
    generation: u64,
    revoke_on_call: AtomicUsize,
    calls: AtomicUsize,
}
impl SwitchWorkerCurrentness {
    fn active(generation: u64) -> Self {
        Self {
            generation,
            revoke_on_call: AtomicUsize::new(usize::MAX),
            calls: AtomicUsize::new(0),
        }
    }
    fn reset_revoke_on(&self, call: usize) {
        self.calls.store(0, Ordering::SeqCst);
        self.revoke_on_call.store(call, Ordering::SeqCst);
    }
}
impl WorkerQualificationCurrentnessSource for SwitchWorkerCurrentness {
    fn current_context(&self, _worker_sha256: [u8; 32]) -> Option<WorkerQualificationContext> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if call >= self.revoke_on_call.load(Ordering::SeqCst) {
            Some(WorkerQualificationContext::revoked(self.generation))
        } else {
            Some(WorkerQualificationContext::active(self.generation))
        }
    }
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples/extensions/hello-simulation")
}

fn exact_subject() -> (Vec<u8>, ExtensionManifest, Vec<u8>) {
    let fixture = fixture_dir();
    let manifest_bytes = fs::read(fixture.join("manifest.json")).unwrap();
    let manifest = serde_json::from_slice(&manifest_bytes).unwrap();
    let component = fs::read(
        fixture.join("target/wasm32-wasip2/release/symthaea_hello_simulation_extension.wasm"),
    )
    .unwrap();
    (manifest_bytes, manifest, component)
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
        PrincipalId::new("test.public-worker-deployment").unwrap(),
        None,
        TrustLevel::Community,
        vec![CapabilityId::new("engineering.simulation.custom")],
        PermissionSet::default(),
        generation,
        trust_generation,
    )
    .unwrap()
}

fn request(id: &str) -> SimulationRequest {
    SimulationRequest::new(
        id,
        EngineeringDomain::Systems,
        SolverKind::Custom,
        "execute exact admitted Component through frozen public worker deployment",
    )
    .with_parameter("alpha", 2.5, "1", "public deployment")
    .with_parameter("beta", 1.5, "1", "public deployment")
}

fn decode_hex_32(value: &str) -> [u8; 32] {
    assert_eq!(value.len(), 64);
    let mut out = [0u8; 32];
    for (slot, pair) in out.iter_mut().zip(value.as_bytes().chunks_exact(2)) {
        *slot = u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap();
    }
    out
}

fn worker_path() -> String {
    std::env::var("SYMTHAEA_PUBLIC_DEPLOYMENT_WORKER_BIN")
        .expect("workflow must provide exact filesystem-contained worker")
}

fn qualification_root() -> [u8; 32] {
    let root = decode_hex_32(
        &std::env::var("SYMTHAEA_WORKER_QUALIFICATION_EVIDENCE_SHA256")
            .expect("workflow must provide qualification evidence root"),
    );
    assert_ne!(root, [0; 32]);
    root
}

fn create_lease(suffix: &str) -> CgroupV2Lease {
    let root = std::env::var("SYMTHAEA_CGROUP_V2_DELEGATED_ROOT")
        .expect("workflow must provide explicit cgroup-v2 resource root");
    CgroupV2Lease::create(
        root,
        &format!("symthaea-public-deploy-{}-{suffix}", std::process::id()),
        public_cgroup_limits_v1(),
    )
    .unwrap()
}

fn cleanup(lease: CgroupV2Lease) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while lease.populated().unwrap() {
        if Instant::now() >= deadline {
            let _ = lease.kill_all();
            panic!("public deployment cgroup remained populated");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    lease.remove_empty().unwrap();
}

struct Subject {
    manifest_bytes: Vec<u8>,
    manifest: ExtensionManifest,
    component: Vec<u8>,
    admission_currentness: SwitchAdmissionCurrentness,
    worker_currentness: SwitchWorkerCurrentness,
    admission_record: AdmissionRecord,
    qualification_record: WorkerQualificationRecord,
}

fn subject() -> Subject {
    let (manifest_bytes, manifest, component) = exact_subject();
    let admission_currentness = SwitchAdmissionCurrentness::active(21, 34);
    let worker_currentness = SwitchWorkerCurrentness::active(55);
    let admission_record = admission_for(&manifest, &manifest_bytes, &component, 21, 34);
    let image = SealedWorkerImage::from_path(worker_path()).unwrap();
    let qualification_record =
        WorkerQualificationRecord::issue(image.image_sha256(), qualification_root(), 55).unwrap();
    Subject {
        manifest_bytes,
        manifest,
        component,
        admission_currentness,
        worker_currentness,
        admission_record,
        qualification_record,
    }
}

fn issue(
    subject: &Subject,
) -> (
    symthaea_extension_admission::ActiveAdmission,
    symthaea_sim_worker_qualification::ActiveWorkerQualification,
    BoundSimulationDeployment,
    BoundPublicWorkerDeployment,
) {
    let admission = subject
        .admission_record
        .activate(&subject.manifest, &subject.admission_currentness)
        .unwrap();
    let qualification_image = SealedWorkerImage::from_path(worker_path()).unwrap();
    let qualification = subject
        .qualification_record
        .activate(&qualification_image, &subject.worker_currentness)
        .unwrap();
    let base = BoundSimulationDeployment::issue(
        &admission,
        &subject.admission_currentness,
        &qualification,
        &subject.worker_currentness,
        subject.manifest_bytes.clone(),
        subject.component.clone(),
        SealedWorkerImage::from_path(worker_path()).unwrap(),
        public_supervisor_limits_v1(),
        public_worker_frame_limits_v1(),
    )
    .unwrap();
    let deployment = BoundPublicWorkerDeployment::issue(
        &base,
        &admission,
        &subject.admission_currentness,
        &qualification,
        &subject.worker_currentness,
        subject.manifest_bytes.clone(),
        subject.component.clone(),
        SealedWorkerImage::from_path(worker_path()).unwrap(),
    )
    .unwrap();
    (admission, qualification, base, deployment)
}

#[test]
#[ignore = "requires explicit cgroup root, exact filesystem worker, qualification root and release-built Component"]
fn exact_public_deployment_executes_with_frozen_profile() {
    let subject = subject();
    let (admission, qualification, base, deployment) = issue(&subject);
    let issued = deployment.evidence();
    issued.verify().unwrap();
    assert_eq!(issued.profile, PUBLIC_WORKER_DEPLOYMENT_PROFILE_V1);
    assert!(issued.legacy_preexec_rlimits_applied);
    assert!(issued.live_worker_qualification_scope_bound);

    let lease = create_lease("success");
    let invocation = deployment
        .execute(
            &base,
            &admission,
            &subject.admission_currentness,
            &qualification,
            &subject.worker_currentness,
            &lease,
            &request("public-deployment-success"),
        )
        .unwrap();
    invocation.evidence().verify().unwrap();
    assert_eq!(invocation.base_deployment_sha256(), base.deployment_sha256());
    assert_eq!(invocation.worker_qualification_evidence_sha256(), qualification_root());
    assert_eq!(
        invocation.evidence().technical.profile,
        PUBLIC_WORKER_EXECUTION_PROFILE_V1
    );
    let result = invocation.result();
    assert_eq!(result.evidence, SimulationEvidence::default());
    assert_eq!(result.metrics.len(), 1);
    assert_eq!(result.metrics[0].name, "fixture.parameter-sum");
    assert_eq!(result.metrics[0].value, 4.0);
    assert_eq!(result.warnings.len(), 1);
    cleanup(lease);
}

#[test]
#[ignore = "requires explicit cgroup root, exact filesystem worker, qualification root and release-built Component"]
fn admission_revoked_after_compute_withholds_public_invocation() {
    let subject = subject();
    let (admission, qualification, base, deployment) = issue(&subject);
    subject.admission_currentness.reset_revoke_on(2);
    subject.worker_currentness.reset_revoke_on(usize::MAX);
    let lease = create_lease("admission-revoke");
    let error = deployment
        .execute(
            &base,
            &admission,
            &subject.admission_currentness,
            &qualification,
            &subject.worker_currentness,
            &lease,
            &request("public-deployment-admission-revoke"),
        )
        .expect_err("post-compute admission revocation must withhold invocation");
    assert!(matches!(
        error,
        PublicWorkerDeploymentError::PostExecutionAdmissionCurrentness(_)
    ));
    assert_eq!(subject.admission_currentness.calls.load(Ordering::SeqCst), 2);
    cleanup(lease);
}

#[test]
#[ignore = "requires explicit cgroup root, exact filesystem worker, qualification root and release-built Component"]
fn worker_revoked_after_compute_withholds_public_invocation() {
    let subject = subject();
    let (admission, qualification, base, deployment) = issue(&subject);
    subject.admission_currentness.reset_revoke_on(usize::MAX);
    subject.worker_currentness.reset_revoke_on(2);
    let lease = create_lease("worker-revoke");
    let error = deployment
        .execute(
            &base,
            &admission,
            &subject.admission_currentness,
            &qualification,
            &subject.worker_currentness,
            &lease,
            &request("public-deployment-worker-revoke"),
        )
        .expect_err("post-compute worker revocation must withhold invocation");
    assert!(matches!(
        error,
        PublicWorkerDeploymentError::PostExecutionWorkerCurrentness(_)
    ));
    assert_eq!(subject.worker_currentness.calls.load(Ordering::SeqCst), 2);
    cleanup(lease);
}
