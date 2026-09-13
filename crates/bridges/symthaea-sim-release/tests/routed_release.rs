use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use symthaea_extension_admission::{
    AdmissionContext, AdmissionCurrentnessSource, AdmissionProblem, AdmissionRecord,
    AdmissionSubject, PrincipalId, Sha256Digest, TrustLevel,
};
use symthaea_extension_core::{CapabilityId, ExtensionManifest, PermissionSet, RuntimeKind};
use symthaea_extension_registry::ExtensionRegistry;
use symthaea_extension_router::{ProviderObservation, ProviderState, RoutingConstraints};
use symthaea_sim_bridge::{EngineeringDomain, ExecutionMode, SimulationRequest, SolverKind};
use symthaea_sim_deployment::BoundSimulationDeployment;
use symthaea_sim_deployment_routing::{
    DeploymentExecutionAuthority, select_routed_deployment,
};
use symthaea_sim_release::{
    CONTAINED_DEPLOYMENT_BACKEND_V1, SimulationReleaseError, release_routed_deployment,
};
use symthaea_sim_worker::{SupervisorLimits, WorkerFrameLimits};
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

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples/extensions/hello-simulation")
}

fn request(id: &str) -> SimulationRequest {
    SimulationRequest::new(
        id,
        EngineeringDomain::Systems,
        SolverKind::Custom,
        "route, contain, execute and canonically release exact Component",
    )
    .with_parameter("alpha", 2.5, "1", "routed release integration")
    .with_parameter("beta", 1.5, "1", "routed release integration")
}

fn admission_record(
    manifest: &ExtensionManifest,
    manifest_bytes: &[u8],
    component_bytes: &[u8],
) -> AdmissionRecord {
    AdmissionRecord::issue(
        manifest.id.clone(),
        manifest.version.clone(),
        Sha256Digest::new(Sha256::digest(manifest_bytes).into()),
        Sha256Digest::new(Sha256::digest(component_bytes).into()),
        Sha256Digest::new([0x77; 32]),
        PrincipalId::new("test.routed-release-authority").unwrap(),
        None,
        TrustLevel::Community,
        vec![CapabilityId::new("engineering.simulation.custom")],
        PermissionSet::default(),
        31,
        37,
    )
    .unwrap()
}

fn observation(manifest: &ExtensionManifest) -> ProviderObservation {
    ProviderObservation {
        extension: manifest.id.clone(),
        state: ProviderState::Ready,
        evidence_grade: 4,
        reliability_bps: 9_950,
        estimated_latency_ms: Some(5),
    }
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

struct FixtureAuthority {
    manifest: ExtensionManifest,
    manifest_bytes: Vec<u8>,
    component: Vec<u8>,
}

fn load_fixture() -> FixtureAuthority {
    let dir = fixture_dir();
    let manifest_bytes = fs::read(dir.join("manifest.json")).unwrap();
    let manifest = serde_json::from_slice(&manifest_bytes).unwrap();
    let component = fs::read(
        dir.join("target/wasm32-wasip2/release/symthaea_hello_simulation_extension.wasm"),
    )
    .unwrap();
    FixtureAuthority {
        manifest,
        manifest_bytes,
        component,
    }
}

#[test]
#[ignore = "requires exact filesystem-contained worker and release-built hello-simulation Component"]
fn routed_contained_component_mints_existing_canonical_release_receipt() {
    let fixture = load_fixture();
    let activation_source = FixedAdmissionCurrentness(AdmissionContext::active(31, 37));
    let admission = admission_record(
        &fixture.manifest,
        &fixture.manifest_bytes,
        &fixture.component,
    )
    .activate(&fixture.manifest, &activation_source)
    .unwrap();

    let worker_path = std::env::var("SYMTHAEA_FILESYSTEM_WORKER_BIN").unwrap();
    let image = SealedWorkerImage::from_path(worker_path).unwrap();
    let qualification_record =
        WorkerQualificationRecord::issue(image.image_sha256(), qualification_root(), 41).unwrap();
    let worker_source = FixedWorkerCurrentness(WorkerQualificationContext::active(41));
    let qualification = qualification_record.activate(&image, &worker_source).unwrap();
    let deployment = BoundSimulationDeployment::issue(
        &admission,
        &activation_source,
        &qualification,
        &worker_source,
        fixture.manifest_bytes.clone(),
        fixture.component.clone(),
        image,
        SupervisorLimits::default(),
        WorkerFrameLimits::default(),
    )
    .unwrap();

    let mut registry = ExtensionRegistry::new();
    registry.register(fixture.manifest.clone()).unwrap();
    let observations = vec![observation(&fixture.manifest)];
    let authorities = vec![DeploymentExecutionAuthority::new(&deployment, &qualification)];
    let admissions = vec![admission];
    let invocation_currentness = FixedAdmissionCurrentness(AdmissionContext::active(31, 37));
    let routed_request = request("real-routed-release");
    let mut constraints = RoutingConstraints::default();
    constraints.allowed_runtimes = vec![RuntimeKind::Wasm];

    let selection = select_routed_deployment(
        &registry,
        &routed_request,
        constraints,
        &admissions,
        &observations,
        &authorities,
        &invocation_currentness,
        &worker_source,
    )
    .unwrap();
    let completed = selection
        .execute(&routed_request, &invocation_currentness, &worker_source)
        .unwrap();
    let release = release_routed_deployment(
        &routed_request,
        &completed,
        &deployment,
        &qualification,
        &worker_source,
        &fixture.manifest,
        &admissions,
        &invocation_currentness,
    )
    .unwrap();

    release.verify(&routed_request).unwrap();
    assert_eq!(release.receipt().runtime(), RuntimeKind::Wasm);
    assert_ne!(release.receipt().release_sha256(), [0; 32]);
    assert_eq!(release.result().evidence.mode, ExecutionMode::ExtensionComponent);
    assert_eq!(
        release.result().evidence.backend.as_deref(),
        Some(CONTAINED_DEPLOYMENT_BACKEND_V1)
    );
    assert!(!release.result().is_engineering_evidence());
    let extension = release.result().evidence.extension.as_ref().unwrap();
    assert!(extension.runtime_profile.contains("deployment="));
    assert!(extension.adapter_version.contains("qualification-evidence="));
    assert_eq!(release.result().metrics.len(), 1);
    assert_eq!(release.result().metrics[0].name, "fixture.parameter-sum");
    assert_eq!(release.result().metrics[0].value, 4.0);
    assert_eq!(release.result().warnings.len(), 1);
    assert!(release.result().warnings[0].contains("not engineering evidence"));
}

#[test]
#[ignore = "requires exact filesystem-contained worker and release-built hello-simulation Component"]
fn admission_revocation_at_release_finalization_withholds_contained_result() {
    let fixture = load_fixture();
    let activation_source = FixedAdmissionCurrentness(AdmissionContext::active(31, 37));
    let admission = admission_record(
        &fixture.manifest,
        &fixture.manifest_bytes,
        &fixture.component,
    )
    .activate(&fixture.manifest, &activation_source)
    .unwrap();

    let worker_path = std::env::var("SYMTHAEA_FILESYSTEM_WORKER_BIN").unwrap();
    let image = SealedWorkerImage::from_path(worker_path).unwrap();
    let qualification_record =
        WorkerQualificationRecord::issue(image.image_sha256(), qualification_root(), 41).unwrap();
    let worker_source = FixedWorkerCurrentness(WorkerQualificationContext::active(41));
    let qualification = qualification_record.activate(&image, &worker_source).unwrap();
    let deployment = BoundSimulationDeployment::issue(
        &admission,
        &activation_source,
        &qualification,
        &worker_source,
        fixture.manifest_bytes.clone(),
        fixture.component.clone(),
        image,
        SupervisorLimits::default(),
        WorkerFrameLimits::default(),
    )
    .unwrap();

    let mut registry = ExtensionRegistry::new();
    registry.register(fixture.manifest.clone()).unwrap();
    let observations = vec![observation(&fixture.manifest)];
    let authorities = vec![DeploymentExecutionAuthority::new(&deployment, &qualification)];
    let admissions = vec![admission];
    // Selection = call 1, deployment pre-execution = call 2, deployment
    // post-execution = call 3, release finalization = call 4 (revoked).
    let currentness = RevokeOnAdmissionCall::new(4, 31, 37);
    let routed_request = request("revoke-admission-at-release");
    let mut constraints = RoutingConstraints::default();
    constraints.allowed_runtimes = vec![RuntimeKind::Wasm];

    let selection = select_routed_deployment(
        &registry,
        &routed_request,
        constraints,
        &admissions,
        &observations,
        &authorities,
        &currentness,
        &worker_source,
    )
    .unwrap();
    let completed = selection
        .execute(&routed_request, &currentness, &worker_source)
        .expect("technical execution and post-execution currentness still pass");
    assert_eq!(currentness.calls(), 3);

    let error = release_routed_deployment(
        &routed_request,
        &completed,
        &deployment,
        &qualification,
        &worker_source,
        &fixture.manifest,
        &admissions,
        &currentness,
    )
    .expect_err("fourth admission currentness check must withhold release");
    assert!(matches!(
        error,
        SimulationReleaseError::ReleaseCurrentness(AdmissionProblem::Revoked)
    ));
    assert_eq!(currentness.calls(), 4);
}

#[test]
#[ignore = "requires exact filesystem-contained worker and release-built hello-simulation Component"]
fn worker_qualification_revocation_at_release_finalization_withholds_contained_result() {
    let fixture = load_fixture();
    let admission_source = FixedAdmissionCurrentness(AdmissionContext::active(31, 37));
    let admission = admission_record(
        &fixture.manifest,
        &fixture.manifest_bytes,
        &fixture.component,
    )
    .activate(&fixture.manifest, &admission_source)
    .unwrap();

    let worker_path = std::env::var("SYMTHAEA_FILESYSTEM_WORKER_BIN").unwrap();
    let image = SealedWorkerImage::from_path(worker_path).unwrap();
    let qualification_record =
        WorkerQualificationRecord::issue(image.image_sha256(), qualification_root(), 41).unwrap();
    let activation_worker_source =
        FixedWorkerCurrentness(WorkerQualificationContext::active(41));
    let qualification = qualification_record
        .activate(&image, &activation_worker_source)
        .unwrap();
    let deployment = BoundSimulationDeployment::issue(
        &admission,
        &admission_source,
        &qualification,
        &activation_worker_source,
        fixture.manifest_bytes.clone(),
        fixture.component.clone(),
        image,
        SupervisorLimits::default(),
        WorkerFrameLimits::default(),
    )
    .unwrap();

    let mut registry = ExtensionRegistry::new();
    registry.register(fixture.manifest.clone()).unwrap();
    let observations = vec![observation(&fixture.manifest)];
    let authorities = vec![DeploymentExecutionAuthority::new(&deployment, &qualification)];
    let admissions = vec![admission];
    // Selection = worker call 1, deployment pre-execution = call 2, deployment
    // post-execution = call 3, release finalization = call 4 (revoked).
    let worker_currentness = RevokeOnWorkerCall::new(4, 41);
    let routed_request = request("revoke-worker-at-release");
    let mut constraints = RoutingConstraints::default();
    constraints.allowed_runtimes = vec![RuntimeKind::Wasm];

    let selection = select_routed_deployment(
        &registry,
        &routed_request,
        constraints,
        &admissions,
        &observations,
        &authorities,
        &admission_source,
        &worker_currentness,
    )
    .unwrap();
    let completed = selection
        .execute(&routed_request, &admission_source, &worker_currentness)
        .expect("technical execution and worker post-execution currentness still pass");
    assert_eq!(worker_currentness.calls(), 3);

    let error = release_routed_deployment(
        &routed_request,
        &completed,
        &deployment,
        &qualification,
        &worker_currentness,
        &fixture.manifest,
        &admissions,
        &admission_source,
    )
    .expect_err("fourth worker qualification currentness check must withhold release");
    assert!(matches!(
        error,
        SimulationReleaseError::ReleaseWorkerCurrentness(WorkerQualificationError::Revoked)
    ));
    assert_eq!(worker_currentness.calls(), 4);
}
