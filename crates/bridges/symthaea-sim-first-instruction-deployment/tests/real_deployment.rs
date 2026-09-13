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
use symthaea_sim_first_instruction_deployment::{
    BoundFirstInstructionDeployment, FirstInstructionDeploymentError, FirstInstructionResourcePolicy,
    FIRST_INSTRUCTION_DEPLOYMENT_PROFILE_V1,
};
use symthaea_sim_first_instruction_worker::FirstInstructionWorkerLimits;
use symthaea_sim_worker::{SupervisorLimits, WorkerFrameLimits};
use symthaea_sim_worker_cgroup::{CgroupV2Lease, CgroupV2Limits};
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
            Some(AdmissionContext::revoked(
                self.generation,
                self.trust_generation,
            ))
        } else {
            Some(AdmissionContext::active(
                self.generation,
                self.trust_generation,
            ))
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
    let manifest_bytes = fs::read(fixture.join("manifest.json")).expect("read fixture manifest");
    let manifest = serde_json::from_slice(&manifest_bytes).expect("parse fixture manifest");
    let component = fs::read(
        fixture.join("target/wasm32-wasip2/release/symthaea_hello_simulation_extension.wasm"),
    )
    .expect("read release Component");
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
        PrincipalId::new("test.first-instruction-deployment").unwrap(),
        None,
        TrustLevel::Community,
        vec![CapabilityId::new("engineering.simulation.custom")],
        PermissionSet::default(),
        generation,
        trust_generation,
    )
    .expect("issue admission")
}

fn request(id: &str) -> SimulationRequest {
    SimulationRequest::new(
        id,
        EngineeringDomain::Systems,
        SolverKind::Custom,
        "execute exact admitted Component through first-instruction resource deployment",
    )
    .with_parameter("alpha", 2.5, "1", "first-instruction deployment")
    .with_parameter("beta", 1.5, "1", "first-instruction deployment")
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
    std::env::var("SYMTHAEA_FILESYSTEM_WORKER_BIN")
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
        &format!("symthaea-first-deploy-{}-{suffix}", std::process::id()),
        CgroupV2Limits::default(),
    )
    .expect("create exact cgroup lease")
}

fn cleanup(lease: CgroupV2Lease) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while lease.populated().expect("read cgroup population") {
        if Instant::now() >= deadline {
            let _ = lease.kill_all();
            panic!("first-instruction deployment cgroup remained populated");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    lease.remove_empty().expect("remove empty cgroup leaf");
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
    let image = SealedWorkerImage::from_path(worker_path()).expect("seal qualification image");
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

#[test]
#[ignore = "requires explicit cgroup root, exact filesystem worker, and release-built Component"]
fn exact_base_deployment_executes_through_first_instruction_resource_binding() {
    let subject = subject();
    let admission = subject
        .admission_record
        .activate(&subject.manifest, &subject.admission_currentness)
        .unwrap();
    let qualification_image = SealedWorkerImage::from_path(worker_path()).unwrap();
    let qualification = subject
        .qualification_record
        .activate(&qualification_image, &subject.worker_currentness)
        .unwrap();

    let supervisor = SupervisorLimits::default();
    let frame = WorkerFrameLimits::default();
    let base = BoundSimulationDeployment::issue(
        &admission,
        &subject.admission_currentness,
        &qualification,
        &subject.worker_currentness,
        subject.manifest_bytes.clone(),
        subject.component.clone(),
        SealedWorkerImage::from_path(worker_path()).unwrap(),
        supervisor,
        frame,
    )
    .unwrap();

    let policy = FirstInstructionResourcePolicy {
        cgroup: CgroupV2Limits::default(),
        worker: FirstInstructionWorkerLimits {
            wall_time_ms: base.evidence().supervisor.wall_time_ms,
            max_stdout_bytes: base.evidence().supervisor.max_stdout_bytes,
            max_stderr_bytes: base.evidence().supervisor.max_stderr_bytes,
        },
    };
    let deployment = BoundFirstInstructionDeployment::issue(
        &base,
        &admission,
        &subject.admission_currentness,
        &qualification,
        &subject.worker_currentness,
        subject.manifest_bytes.clone(),
        subject.component.clone(),
        SealedWorkerImage::from_path(worker_path()).unwrap(),
        policy,
    )
    .unwrap();
    let issued = deployment.evidence();
    issued.verify().unwrap();
    assert_eq!(issued.profile, FIRST_INSTRUCTION_DEPLOYMENT_PROFILE_V1);
    assert!(!issued.legacy_preexec_rlimits_applied);
    assert!(issued.containment_claim_requires_live_worker_qualification);

    let lease = create_lease("success");
    let invocation = deployment
        .execute(
            &base,
            &admission,
            &subject.admission_currentness,
            &qualification,
            &subject.worker_currentness,
            &lease,
            &request("first-instruction-deployment-success"),
        )
        .unwrap();
    invocation.evidence().verify().unwrap();
    assert_eq!(invocation.base_deployment_sha256(), base.deployment_sha256());
    assert_eq!(
        invocation.worker_qualification_evidence_sha256(),
        qualification_root()
    );
    let result = invocation.result();
    assert_eq!(result.evidence, SimulationEvidence::default());
    assert_eq!(result.metrics.len(), 1);
    assert_eq!(result.metrics[0].name, "fixture.parameter-sum");
    assert_eq!(result.metrics[0].value, 4.0);
    cleanup(lease);
}

#[test]
#[ignore = "requires explicit cgroup root, exact filesystem worker, and release-built Component"]
fn admission_revoked_after_compute_withholds_first_instruction_invocation() {
    let subject = subject();
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
        SupervisorLimits::default(),
        WorkerFrameLimits::default(),
    )
    .unwrap();
    let base_evidence = base.evidence();
    let deployment = BoundFirstInstructionDeployment::issue(
        &base,
        &admission,
        &subject.admission_currentness,
        &qualification,
        &subject.worker_currentness,
        subject.manifest_bytes.clone(),
        subject.component.clone(),
        SealedWorkerImage::from_path(worker_path()).unwrap(),
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
            &request("first-instruction-deployment-admission-revoke"),
        )
        .expect_err("post-compute admission revocation must withhold invocation");
    assert!(matches!(
        error,
        FirstInstructionDeploymentError::PostExecutionAdmissionCurrentness(_)
    ));
    assert_eq!(subject.admission_currentness.calls.load(Ordering::SeqCst), 2);
    cleanup(lease);
}

#[test]
#[ignore = "requires explicit cgroup root, exact filesystem worker, and release-built Component"]
fn worker_qualification_revoked_after_compute_withholds_first_instruction_invocation() {
    let subject = subject();
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
        SupervisorLimits::default(),
        WorkerFrameLimits::default(),
    )
    .unwrap();
    let base_evidence = base.evidence();
    let deployment = BoundFirstInstructionDeployment::issue(
        &base,
        &admission,
        &subject.admission_currentness,
        &qualification,
        &subject.worker_currentness,
        subject.manifest_bytes.clone(),
        subject.component.clone(),
        SealedWorkerImage::from_path(worker_path()).unwrap(),
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
            &request("first-instruction-deployment-worker-revoke"),
        )
        .expect_err("post-compute worker revocation must withhold invocation");
    assert!(matches!(
        error,
        FirstInstructionDeploymentError::PostExecutionWorkerCurrentness(_)
    ));
    assert_eq!(subject.worker_currentness.calls.load(Ordering::SeqCst), 2);
    cleanup(lease);
}
