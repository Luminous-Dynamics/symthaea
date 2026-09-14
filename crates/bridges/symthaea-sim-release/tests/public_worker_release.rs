use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use symthaea_extension_admission::{
    AdmissionContext, AdmissionCurrentnessSource, AdmissionRecord, AdmissionSubject, PrincipalId,
    Sha256Digest, TrustLevel,
};
use symthaea_extension_core::{CapabilityId, ExtensionManifest, PermissionSet, RuntimeKind};
use symthaea_extension_registry::ExtensionRegistry;
use symthaea_extension_router::{ProviderObservation, ProviderState, RoutingConstraints};
use symthaea_sim_bridge::{EngineeringDomain, ExecutionMode, SimulationRequest, SolverKind};
use symthaea_sim_deployment::BoundSimulationDeployment;
use symthaea_sim_public_worker_deployment::BoundPublicWorkerDeployment;
use symthaea_sim_public_worker_execution::{public_cgroup_limits_v1, public_supervisor_limits_v1};
use symthaea_sim_public_worker_input::public_worker_frame_limits_v1;
use symthaea_sim_public_worker_routing::{
    PublicWorkerExecutionAuthority, select_routed_public_worker,
};
use symthaea_sim_release::{
    PUBLIC_WORKER_DEPLOYMENT_BACKEND_V1, SIMULATION_RELEASE_PROFILE_V1, SimulationReleaseError,
    release_routed_public_worker,
};
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
    let manifest_bytes = fs::read(fixture.join("manifest.json")).unwrap();
    let manifest = serde_json::from_slice(&manifest_bytes).unwrap();
    let component = fs::read(
        fixture.join("target/wasm32-wasip2/release/symthaea_hello_simulation_extension.wasm"),
    )
    .unwrap();
    (manifest_bytes, manifest, component)
}

fn request(id: &str) -> SimulationRequest {
    SimulationRequest::new(
        id,
        EngineeringDomain::Systems,
        SolverKind::Custom,
        "route and release exact frozen public Component",
    )
    .with_parameter("alpha", 2.5, "1", "public release")
    .with_parameter("beta", 1.5, "1", "public release")
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
    std::env::var("SYMTHAEA_PUBLIC_RELEASE_WORKER_BIN")
        .expect("workflow provides exact filesystem-contained worker")
}

fn qualification_root() -> [u8; 32] {
    let root = decode_hex_32(
        &std::env::var("SYMTHAEA_WORKER_QUALIFICATION_EVIDENCE_SHA256")
            .expect("workflow provides worker qualification root"),
    );
    assert_ne!(root, [0; 32]);
    root
}

fn create_lease(suffix: &str) -> CgroupV2Lease {
    let root = std::env::var("SYMTHAEA_CGROUP_V2_DELEGATED_ROOT")
        .expect("workflow provides explicit cgroup-v2 resource root");
    CgroupV2Lease::create(
        root,
        &format!("symthaea-public-release-{}-{suffix}", std::process::id()),
        public_cgroup_limits_v1(),
    )
    .unwrap()
}

fn cleanup(lease: CgroupV2Lease) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while lease.populated().unwrap() {
        if Instant::now() >= deadline {
            let _ = lease.kill_all();
            panic!("public release cgroup remained populated");
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
    let admission_record = AdmissionRecord::issue(
        manifest.id.clone(),
        manifest.version.clone(),
        Sha256Digest::new(Sha256::digest(&manifest_bytes).into()),
        Sha256Digest::new(Sha256::digest(&component).into()),
        Sha256Digest::new([0x55; 32]),
        PrincipalId::new("test.public-worker-release").unwrap(),
        None,
        TrustLevel::Community,
        vec![CapabilityId::new("engineering.simulation.custom")],
        PermissionSet::default(),
        21,
        34,
    )
    .unwrap();
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

fn observation(manifest: &ExtensionManifest) -> ProviderObservation {
    ProviderObservation {
        extension: manifest.id.clone(),
        state: ProviderState::Ready,
        evidence_grade: 5,
        reliability_bps: 10_000,
        estimated_latency_ms: Some(1),
    }
}

#[allow(clippy::type_complexity)]
fn prepare(
    subject: &Subject,
) -> (
    symthaea_extension_admission::ActiveAdmission,
    symthaea_sim_worker_qualification::ActiveWorkerQualification,
    BoundSimulationDeployment,
    BoundPublicWorkerDeployment,
    ExtensionRegistry,
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
    let mut registry = ExtensionRegistry::new();
    registry.register(subject.manifest.clone()).unwrap();
    (admission, qualification, base, deployment, registry)
}

#[allow(clippy::too_many_arguments)]
fn route_and_execute(
    subject: &Subject,
    admission: &symthaea_extension_admission::ActiveAdmission,
    qualification: &symthaea_sim_worker_qualification::ActiveWorkerQualification,
    base: &BoundSimulationDeployment,
    deployment: &BoundPublicWorkerDeployment,
    registry: &ExtensionRegistry,
    request: &SimulationRequest,
    lease: &CgroupV2Lease,
) -> symthaea_sim_public_worker_routing::RoutedPublicWorkerInvocation {
    let admissions = std::slice::from_ref(admission);
    let observations = [observation(&subject.manifest)];
    let authorities = [PublicWorkerExecutionAuthority::new(base, deployment, qualification)];
    let selected = select_routed_public_worker(
        registry,
        request,
        RoutingConstraints::default(),
        admissions,
        &observations,
        &authorities,
        &subject.admission_currentness,
        &subject.worker_currentness,
    )
    .unwrap();
    selected
        .execute(
            request,
            &subject.admission_currentness,
            &subject.worker_currentness,
            lease,
        )
        .unwrap()
}

#[test]
#[ignore = "requires explicit cgroup root, exact filesystem worker, qualification root and release-built Component"]
fn public_route_executes_and_mints_existing_release_v1() {
    let subject = subject();
    let (admission, qualification, base, deployment, registry) = prepare(&subject);
    subject.admission_currentness.reset_revoke_on(usize::MAX);
    subject.worker_currentness.reset_revoke_on(usize::MAX);
    let request = request("public-release-success");
    let lease = create_lease("success");
    let routed = route_and_execute(
        &subject,
        &admission,
        &qualification,
        &base,
        &deployment,
        &registry,
        &request,
        &lease,
    );
    let admissions = [admission];
    let released = release_routed_public_worker(
        &request,
        &routed,
        &base,
        &deployment,
        &qualification,
        &subject.worker_currentness,
        &subject.manifest,
        &admissions,
        &subject.admission_currentness,
    )
    .unwrap();
    released.verify(&request).unwrap();
    assert_eq!(released.evidence().profile, SIMULATION_RELEASE_PROFILE_V1);
    assert_eq!(released.receipt().runtime(), RuntimeKind::Wasm);
    assert_eq!(released.result().evidence.mode, ExecutionMode::ExtensionComponent);
    assert_eq!(
        released.result().evidence.backend.as_deref(),
        Some(PUBLIC_WORKER_DEPLOYMENT_BACKEND_V1)
    );
    assert!(!released.result().is_engineering_evidence());
    assert_eq!(released.result().metrics.len(), 1);
    assert_eq!(released.result().metrics[0].name, "fixture.parameter-sum");
    assert_eq!(released.result().metrics[0].value, 4.0);
    let extension = released.result().evidence.extension.as_ref().unwrap();
    assert!(extension.runtime_profile.contains("public-execution-profile="));
    assert!(extension.runtime_profile.contains("public-deployment="));
    assert!(extension.adapter_version.contains("legacy-preexec-rlimits=true"));
    assert_eq!(subject.admission_currentness.calls.load(Ordering::SeqCst), 4);
    assert_eq!(subject.worker_currentness.calls.load(Ordering::SeqCst), 4);
    cleanup(lease);
}

#[test]
#[ignore = "requires explicit cgroup root, exact filesystem worker, qualification root and release-built Component"]
fn admission_revoked_at_final_release_withholds_public_receipt() {
    let subject = subject();
    let (admission, qualification, base, deployment, registry) = prepare(&subject);
    subject.admission_currentness.reset_revoke_on(4);
    subject.worker_currentness.reset_revoke_on(usize::MAX);
    let request = request("public-release-admission-revoke");
    let lease = create_lease("admission-revoke");
    let routed = route_and_execute(
        &subject,
        &admission,
        &qualification,
        &base,
        &deployment,
        &registry,
        &request,
        &lease,
    );
    let admissions = [admission];
    let error = release_routed_public_worker(
        &request,
        &routed,
        &base,
        &deployment,
        &qualification,
        &subject.worker_currentness,
        &subject.manifest,
        &admissions,
        &subject.admission_currentness,
    )
    .expect_err("final admission revocation must withhold release");
    assert!(matches!(error, SimulationReleaseError::ReleaseCurrentness(_)));
    assert_eq!(subject.admission_currentness.calls.load(Ordering::SeqCst), 4);
    assert_eq!(subject.worker_currentness.calls.load(Ordering::SeqCst), 3);
    cleanup(lease);
}

#[test]
#[ignore = "requires explicit cgroup root, exact filesystem worker, qualification root and release-built Component"]
fn worker_revoked_at_final_release_withholds_public_receipt() {
    let subject = subject();
    let (admission, qualification, base, deployment, registry) = prepare(&subject);
    subject.admission_currentness.reset_revoke_on(usize::MAX);
    subject.worker_currentness.reset_revoke_on(4);
    let request = request("public-release-worker-revoke");
    let lease = create_lease("worker-revoke");
    let routed = route_and_execute(
        &subject,
        &admission,
        &qualification,
        &base,
        &deployment,
        &registry,
        &request,
        &lease,
    );
    let admissions = [admission];
    let error = release_routed_public_worker(
        &request,
        &routed,
        &base,
        &deployment,
        &qualification,
        &subject.worker_currentness,
        &subject.manifest,
        &admissions,
        &subject.admission_currentness,
    )
    .expect_err("final worker revocation must withhold release");
    assert!(matches!(
        error,
        SimulationReleaseError::ReleaseWorkerCurrentness(_)
    ));
    assert_eq!(subject.admission_currentness.calls.load(Ordering::SeqCst), 4);
    assert_eq!(subject.worker_currentness.calls.load(Ordering::SeqCst), 4);
    cleanup(lease);
}
