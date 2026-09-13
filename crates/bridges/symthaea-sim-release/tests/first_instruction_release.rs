use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use symthaea_extension_admission::{
    AdmissionContext, AdmissionCurrentnessSource, AdmissionProblem, AdmissionRecord,
    AdmissionSubject, PrincipalId, Sha256Digest, TrustLevel,
};
use symthaea_extension_core::{CapabilityId, ExtensionManifest, PermissionSet, RuntimeKind};
use symthaea_extension_registry::ExtensionRegistry;
use symthaea_extension_router::{ProviderObservation, ProviderState, RoutingConstraints};
use symthaea_sim_bridge::{EngineeringDomain, ExecutionMode, SimulationRequest, SolverKind};
use symthaea_sim_deployment::BoundSimulationDeployment;
use symthaea_sim_first_instruction_deployment::{
    BoundFirstInstructionDeployment, FirstInstructionResourcePolicy,
};
use symthaea_sim_first_instruction_routing::{
    FirstInstructionExecutionAuthority, select_routed_first_instruction,
};
use symthaea_sim_first_instruction_worker::FirstInstructionWorkerLimits;
use symthaea_sim_release::{
    FIRST_INSTRUCTION_DEPLOYMENT_BACKEND_V1, SimulationReleaseError,
    release_routed_first_instruction,
};
use symthaea_sim_worker::{SupervisorLimits, WorkerFrameLimits};
use symthaea_sim_worker_cgroup::{CgroupV2Lease, CgroupV2Limits};
use symthaea_sim_worker_image::SealedWorkerImage;
use symthaea_sim_worker_qualification::{
    WorkerQualificationContext, WorkerQualificationCurrentnessSource, WorkerQualificationError,
    WorkerQualificationRecord,
};

#[derive(Debug)]
struct FixedAdmissionCurrentness(AdmissionContext);

impl AdmissionCurrentnessSource for FixedAdmissionCurrentness {
    fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
        Some(self.0)
    }
}

#[derive(Debug)]
struct RevokeOnAdmissionCall {
    calls: AtomicUsize,
    revoke_at: usize,
    generation: u64,
    trust_generation: u64,
}

impl RevokeOnAdmissionCall {
    fn new(revoke_at: usize, generation: u64, trust_generation: u64) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            revoke_at,
            generation,
            trust_generation,
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl AdmissionCurrentnessSource for RevokeOnAdmissionCall {
    fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        Some(if call >= self.revoke_at {
            AdmissionContext::revoked(self.generation, self.trust_generation)
        } else {
            AdmissionContext::active(self.generation, self.trust_generation)
        })
    }
}

#[derive(Debug)]
struct FixedWorkerCurrentness(WorkerQualificationContext);

impl WorkerQualificationCurrentnessSource for FixedWorkerCurrentness {
    fn current_context(&self, _worker_sha256: [u8; 32]) -> Option<WorkerQualificationContext> {
        Some(self.0)
    }
}

#[derive(Debug)]
struct RevokeOnWorkerCall {
    calls: AtomicUsize,
    revoke_at: usize,
    generation: u64,
}

impl RevokeOnWorkerCall {
    fn new(revoke_at: usize, generation: u64) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            revoke_at,
            generation,
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl WorkerQualificationCurrentnessSource for RevokeOnWorkerCall {
    fn current_context(&self, _worker_sha256: [u8; 32]) -> Option<WorkerQualificationContext> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        Some(if call >= self.revoke_at {
            WorkerQualificationContext::revoked(self.generation)
        } else {
            WorkerQualificationContext::active(self.generation)
        })
    }
}

struct Fixture {
    manifest: ExtensionManifest,
    manifest_bytes: Vec<u8>,
    component: Vec<u8>,
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples/extensions/hello-simulation")
}

fn load_fixture() -> Fixture {
    let dir = fixture_dir();
    let manifest_bytes = fs::read(dir.join("manifest.json")).unwrap();
    let manifest = serde_json::from_slice(&manifest_bytes).unwrap();
    let component = fs::read(
        dir.join("target/wasm32-wasip2/release/symthaea_hello_simulation_extension.wasm"),
    )
    .unwrap();
    Fixture {
        manifest,
        manifest_bytes,
        component,
    }
}

fn request(id: &str) -> SimulationRequest {
    SimulationRequest::new(
        id,
        EngineeringDomain::Systems,
        SolverKind::Custom,
        "route, execute from cgroup birth, and canonically release exact Component",
    )
    .with_parameter("alpha", 2.5, "1", "first-instruction release")
    .with_parameter("beta", 1.5, "1", "first-instruction release")
}

fn admission_record(fixture: &Fixture) -> AdmissionRecord {
    AdmissionRecord::issue(
        fixture.manifest.id.clone(),
        fixture.manifest.version.clone(),
        Sha256Digest::new(Sha256::digest(&fixture.manifest_bytes).into()),
        Sha256Digest::new(Sha256::digest(&fixture.component).into()),
        Sha256Digest::new([0x91; 32]),
        PrincipalId::new("test.first-instruction-release-authority").unwrap(),
        None,
        TrustLevel::Community,
        vec![CapabilityId::new("engineering.simulation.custom")],
        PermissionSet::default(),
        61,
        67,
    )
    .unwrap()
}

fn observation(manifest: &ExtensionManifest) -> ProviderObservation {
    ProviderObservation {
        extension: manifest.id.clone(),
        state: ProviderState::Ready,
        evidence_grade: 4,
        reliability_bps: 9_970,
        estimated_latency_ms: Some(5),
    }
}

fn worker_path() -> String {
    std::env::var("SYMTHAEA_FILESYSTEM_WORKER_BIN")
        .expect("workflow provides exact filesystem-contained worker")
}

fn qualification_root() -> [u8; 32] {
    let value = std::env::var("SYMTHAEA_WORKER_QUALIFICATION_EVIDENCE_SHA256")
        .expect("workflow provides exact worker qualification evidence root");
    assert_eq!(value.len(), 64);
    let mut bytes = [0u8; 32];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        bytes[index] =
            u8::from_str_radix(std::str::from_utf8(chunk).unwrap(), 16).unwrap();
    }
    assert_ne!(bytes, [0; 32]);
    bytes
}

fn create_lease(suffix: &str) -> CgroupV2Lease {
    let root = std::env::var("SYMTHAEA_CGROUP_V2_DELEGATED_ROOT")
        .expect("workflow provides explicit cgroup-v2 resource root");
    CgroupV2Lease::create(
        root,
        &format!("symthaea-first-release-{}-{suffix}", std::process::id()),
        CgroupV2Limits::default(),
    )
    .unwrap()
}

fn cleanup(lease: CgroupV2Lease) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while lease.populated().unwrap() {
        if Instant::now() >= deadline {
            let _ = lease.kill_all();
            panic!("first-instruction release cgroup remained populated");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    lease.remove_empty().unwrap();
}

struct AuthoritySet {
    admission: symthaea_extension_admission::ActiveAdmission,
    qualification: symthaea_sim_worker_qualification::ActiveWorkerQualification,
    base: BoundSimulationDeployment,
    first: BoundFirstInstructionDeployment,
}

fn authorities(
    fixture: &Fixture,
    admission_source: &dyn AdmissionCurrentnessSource,
    worker_source: &dyn WorkerQualificationCurrentnessSource,
) -> AuthoritySet {
    let activation_admission = FixedAdmissionCurrentness(AdmissionContext::active(61, 67));
    let admission = admission_record(fixture)
        .activate(&fixture.manifest, &activation_admission)
        .unwrap();

    let activation_image = SealedWorkerImage::from_path(worker_path()).unwrap();
    let qualification_record = WorkerQualificationRecord::issue(
        activation_image.image_sha256(),
        qualification_root(),
        71,
    )
    .unwrap();
    let activation_worker = FixedWorkerCurrentness(WorkerQualificationContext::active(71));
    let qualification = qualification_record
        .activate(&activation_image, &activation_worker)
        .unwrap();

    let base = BoundSimulationDeployment::issue(
        &admission,
        admission_source,
        &qualification,
        worker_source,
        fixture.manifest_bytes.clone(),
        fixture.component.clone(),
        SealedWorkerImage::from_path(worker_path()).unwrap(),
        SupervisorLimits::default(),
        WorkerFrameLimits::default(),
    )
    .unwrap();
    let base_evidence = base.evidence();
    let first = BoundFirstInstructionDeployment::issue(
        &base,
        &admission,
        admission_source,
        &qualification,
        worker_source,
        fixture.manifest_bytes.clone(),
        fixture.component.clone(),
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
    AuthoritySet {
        admission,
        qualification,
        base,
        first,
    }
}

#[test]
#[ignore = "requires explicit cgroup root, exact qualified filesystem worker, and release-built Component"]
fn routed_first_instruction_component_mints_existing_release_v1() {
    let fixture = load_fixture();
    let admission_source = FixedAdmissionCurrentness(AdmissionContext::active(61, 67));
    let worker_source = FixedWorkerCurrentness(WorkerQualificationContext::active(71));
    let set = authorities(&fixture, &admission_source, &worker_source);
    let routed_request = request("first-instruction-release-success");
    let lease = create_lease("success");

    let mut registry = ExtensionRegistry::new();
    registry.register(fixture.manifest.clone()).unwrap();
    let observations = vec![observation(&fixture.manifest)];
    let admissions = vec![set.admission];
    let route_authorities = vec![FirstInstructionExecutionAuthority::new(
        &set.base,
        &set.first,
        &set.qualification,
    )];
    let mut constraints = RoutingConstraints::default();
    constraints.allowed_runtimes = vec![RuntimeKind::Wasm];
    let selection = select_routed_first_instruction(
        &registry,
        &routed_request,
        constraints,
        &admissions,
        &observations,
        &route_authorities,
        &admission_source,
        &worker_source,
    )
    .unwrap();
    let completed = selection
        .execute(&routed_request, &admission_source, &worker_source, &lease)
        .unwrap();
    let release = release_routed_first_instruction(
        &routed_request,
        &completed,
        &set.base,
        &set.first,
        &set.qualification,
        &worker_source,
        &fixture.manifest,
        &admissions,
        &admission_source,
    )
    .unwrap();

    release.verify(&routed_request).unwrap();
    assert_eq!(release.receipt().runtime(), RuntimeKind::Wasm);
    assert_ne!(release.receipt().release_sha256(), [0; 32]);
    assert_eq!(release.result().evidence.mode, ExecutionMode::ExtensionComponent);
    assert_eq!(
        release.result().evidence.backend.as_deref(),
        Some(FIRST_INSTRUCTION_DEPLOYMENT_BACKEND_V1)
    );
    assert!(!release.result().is_engineering_evidence());
    let extension = release.result().evidence.extension.as_ref().unwrap();
    assert!(extension.runtime_profile.contains("first-deployment="));
    assert!(extension.runtime_profile.contains("first-execution="));
    assert!(extension.runtime_profile.contains("cgroup="));
    assert!(extension.runtime_profile.contains("sealed-launch="));
    assert!(extension.adapter_version.contains("qualification-evidence="));
    assert!(extension.adapter_version.contains("legacy-preexec-rlimits=false"));
    assert_eq!(release.result().metrics.len(), 1);
    assert_eq!(release.result().metrics[0].name, "fixture.parameter-sum");
    assert_eq!(release.result().metrics[0].value, 4.0);
    assert_eq!(release.result().warnings.len(), 1);
    assert!(release.result().warnings[0].contains("not engineering evidence"));
    cleanup(lease);
}

#[test]
#[ignore = "requires explicit cgroup root, exact qualified filesystem worker, and release-built Component"]
fn admission_revocation_at_final_release_withholds_first_instruction_result() {
    let fixture = load_fixture();
    let setup_admission = FixedAdmissionCurrentness(AdmissionContext::active(61, 67));
    let setup_worker = FixedWorkerCurrentness(WorkerQualificationContext::active(71));
    let set = authorities(&fixture, &setup_admission, &setup_worker);
    let currentness = RevokeOnAdmissionCall::new(4, 61, 67);
    let worker_currentness = FixedWorkerCurrentness(WorkerQualificationContext::active(71));
    let routed_request = request("first-instruction-release-admission-revoke");
    let lease = create_lease("admission-revoke");

    let mut registry = ExtensionRegistry::new();
    registry.register(fixture.manifest.clone()).unwrap();
    let observations = vec![observation(&fixture.manifest)];
    let admissions = vec![set.admission];
    let route_authorities = vec![FirstInstructionExecutionAuthority::new(
        &set.base,
        &set.first,
        &set.qualification,
    )];
    let mut constraints = RoutingConstraints::default();
    constraints.allowed_runtimes = vec![RuntimeKind::Wasm];
    let selection = select_routed_first_instruction(
        &registry,
        &routed_request,
        constraints,
        &admissions,
        &observations,
        &route_authorities,
        &currentness,
        &worker_currentness,
    )
    .unwrap();
    let completed = selection
        .execute(&routed_request, &currentness, &worker_currentness, &lease)
        .expect("route, execution and post-execution admission check pass");
    assert_eq!(currentness.calls(), 3);

    let error = release_routed_first_instruction(
        &routed_request,
        &completed,
        &set.base,
        &set.first,
        &set.qualification,
        &worker_currentness,
        &fixture.manifest,
        &admissions,
        &currentness,
    )
    .expect_err("final admission currentness must withhold release");
    assert!(matches!(
        error,
        SimulationReleaseError::ReleaseCurrentness(AdmissionProblem::Revoked)
    ));
    assert_eq!(currentness.calls(), 4);
    cleanup(lease);
}

#[test]
#[ignore = "requires explicit cgroup root, exact qualified filesystem worker, and release-built Component"]
fn worker_revocation_at_final_release_withholds_first_instruction_result() {
    let fixture = load_fixture();
    let setup_admission = FixedAdmissionCurrentness(AdmissionContext::active(61, 67));
    let setup_worker = FixedWorkerCurrentness(WorkerQualificationContext::active(71));
    let set = authorities(&fixture, &setup_admission, &setup_worker);
    let admission_currentness = FixedAdmissionCurrentness(AdmissionContext::active(61, 67));
    let worker_currentness = RevokeOnWorkerCall::new(4, 71);
    let routed_request = request("first-instruction-release-worker-revoke");
    let lease = create_lease("worker-revoke");

    let mut registry = ExtensionRegistry::new();
    registry.register(fixture.manifest.clone()).unwrap();
    let observations = vec![observation(&fixture.manifest)];
    let admissions = vec![set.admission];
    let route_authorities = vec![FirstInstructionExecutionAuthority::new(
        &set.base,
        &set.first,
        &set.qualification,
    )];
    let mut constraints = RoutingConstraints::default();
    constraints.allowed_runtimes = vec![RuntimeKind::Wasm];
    let selection = select_routed_first_instruction(
        &registry,
        &routed_request,
        constraints,
        &admissions,
        &observations,
        &route_authorities,
        &admission_currentness,
        &worker_currentness,
    )
    .unwrap();
    let completed = selection
        .execute(
            &routed_request,
            &admission_currentness,
            &worker_currentness,
            &lease,
        )
        .expect("route, execution and post-execution worker check pass");
    assert_eq!(worker_currentness.calls(), 3);

    let error = release_routed_first_instruction(
        &routed_request,
        &completed,
        &set.base,
        &set.first,
        &set.qualification,
        &worker_currentness,
        &fixture.manifest,
        &admissions,
        &admission_currentness,
    )
    .expect_err("final worker qualification currentness must withhold release");
    assert!(matches!(
        error,
        SimulationReleaseError::ReleaseWorkerCurrentness(WorkerQualificationError::Revoked)
    ));
    assert_eq!(worker_currentness.calls(), 4);
    cleanup(lease);
}
