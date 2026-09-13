use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use symthaea_extension_admission::{
    AdmissionContext, AdmissionCurrentnessSource, AdmissionRecord, AdmissionSubject, PrincipalId,
    Sha256Digest, TrustLevel,
};
use symthaea_extension_core::{CapabilityId, ExtensionManifest, PermissionSet, RuntimeKind};
use symthaea_extension_registry::ExtensionRegistry;
use symthaea_extension_router::{
    ProviderObservation, ProviderState, RoutingConstraints,
};
use symthaea_sim_bridge::{EngineeringDomain, SimulationEvidence, SimulationRequest, SolverKind};
use symthaea_sim_deployment::BoundSimulationDeployment;
use symthaea_sim_deployment_routing::{
    DeploymentExecutionAuthority, RoutedDeploymentError, select_routed_deployment,
};
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

fn request(id: &str) -> SimulationRequest {
    SimulationRequest::new(
        id,
        EngineeringDomain::Systems,
        SolverKind::Custom,
        "route exact request into exact deployment",
    )
    .with_parameter("alpha", 2.5, "1", "routing integration")
    .with_parameter("beta", 1.5, "1", "routing integration")
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
        Sha256Digest::new([0x66; 32]),
        PrincipalId::new("test.routed-deployment-authority").unwrap(),
        None,
        TrustLevel::Community,
        vec![CapabilityId::new("engineering.simulation.custom")],
        PermissionSet::default(),
        13,
        17,
    )
    .unwrap()
}

fn observation(manifest: &ExtensionManifest) -> ProviderObservation {
    ProviderObservation {
        extension: manifest.id.clone(),
        state: ProviderState::Ready,
        evidence_grade: 3,
        reliability_bps: 9_900,
        estimated_latency_ms: Some(5),
    }
}

#[test]
fn routing_preserves_wasm_manifest_runtime_and_binds_canonical_request() {
    let manifest_bytes = fs::read(fixture_dir().join("manifest.json")).unwrap();
    let manifest: ExtensionManifest = serde_json::from_slice(&manifest_bytes).unwrap();
    let component = b"synthetic-routed-component".to_vec();
    let admission_source = FixedAdmissionCurrentness(AdmissionContext::active(13, 17));
    let admission = admission_record(&manifest, &manifest_bytes, &component)
        .activate(&manifest, &admission_source)
        .unwrap();

    let image = SealedWorkerImage::from_path(std::env::current_exe().unwrap()).unwrap();
    let qualification_record =
        WorkerQualificationRecord::issue(image.image_sha256(), [0x91; 32], 23).unwrap();
    let worker_source = FixedWorkerCurrentness(WorkerQualificationContext::active(23));
    let qualification = qualification_record.activate(&image, &worker_source).unwrap();
    let deployment = BoundSimulationDeployment::issue(
        &admission,
        &admission_source,
        &qualification,
        &worker_source,
        manifest_bytes,
        component,
        image,
        SupervisorLimits::default(),
        WorkerFrameLimits::default(),
    )
    .unwrap();

    let mut registry = ExtensionRegistry::new();
    registry.register(manifest.clone()).unwrap();
    let observations = vec![observation(&manifest)];
    let authorities = vec![DeploymentExecutionAuthority::new(&deployment, &qualification)];
    let admissions = vec![admission];
    let routed_request = request("route-a");

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
        &worker_source,
    )
    .unwrap();

    assert_eq!(selection.decision().selected, manifest.id);
    assert_eq!(registry.get(&selection.decision().selected).unwrap().runtime, RuntimeKind::Wasm);
    assert_eq!(selection.deployment().selected_extension(), "org.example.hello-simulation");
    assert_ne!(selection.request_sha256(), [0; 32]);

    let substituted = request("route-b");
    let error = selection
        .execute(&substituted, &admission_source, &worker_source)
        .expect_err("request substitution must fail before worker spawn");
    assert!(matches!(error, RoutedDeploymentError::RequestSubstitution));
}

#[test]
fn remote_only_constraint_does_not_reinterpret_contained_wasm_as_remote() {
    let manifest_bytes = fs::read(fixture_dir().join("manifest.json")).unwrap();
    let manifest: ExtensionManifest = serde_json::from_slice(&manifest_bytes).unwrap();
    let component = b"synthetic-routed-component".to_vec();
    let admission_source = FixedAdmissionCurrentness(AdmissionContext::active(13, 17));
    let admission = admission_record(&manifest, &manifest_bytes, &component)
        .activate(&manifest, &admission_source)
        .unwrap();

    let image = SealedWorkerImage::from_path(std::env::current_exe().unwrap()).unwrap();
    let qualification_record =
        WorkerQualificationRecord::issue(image.image_sha256(), [0x91; 32], 23).unwrap();
    let worker_source = FixedWorkerCurrentness(WorkerQualificationContext::active(23));
    let qualification = qualification_record.activate(&image, &worker_source).unwrap();
    let deployment = BoundSimulationDeployment::issue(
        &admission,
        &admission_source,
        &qualification,
        &worker_source,
        manifest_bytes,
        component,
        image,
        SupervisorLimits::default(),
        WorkerFrameLimits::default(),
    )
    .unwrap();

    let mut registry = ExtensionRegistry::new();
    registry.register(manifest.clone()).unwrap();
    let observations = vec![observation(&manifest)];
    let authorities = vec![DeploymentExecutionAuthority::new(&deployment, &qualification)];
    let admissions = vec![admission];
    let mut constraints = RoutingConstraints::default();
    constraints.allowed_runtimes = vec![RuntimeKind::Remote];

    let error = select_routed_deployment(
        &registry,
        &request("remote-only"),
        constraints,
        &admissions,
        &observations,
        &authorities,
        &admission_source,
        &worker_source,
    )
    .expect_err("remote-only policy must reject exact wasm manifest");
    assert!(matches!(error, RoutedDeploymentError::Routing(_)));
}

#[test]
#[ignore = "requires exact filesystem-contained worker and release-built hello-simulation Component"]
fn deterministic_route_executes_real_component_through_exact_deployment() {
    let manifest_bytes = fs::read(fixture_dir().join("manifest.json")).unwrap();
    let manifest: ExtensionManifest = serde_json::from_slice(&manifest_bytes).unwrap();
    let component = fs::read(
        fixture_dir().join("target/wasm32-wasip2/release/symthaea_hello_simulation_extension.wasm"),
    )
    .unwrap();
    let qualification_hex = std::env::var("SYMTHAEA_WORKER_QUALIFICATION_EVIDENCE_SHA256")
        .expect("workflow provides qualification root");
    let mut qualification_evidence = [0u8; 32];
    assert_eq!(qualification_hex.len(), 64);
    for (index, chunk) in qualification_hex.as_bytes().chunks_exact(2).enumerate() {
        qualification_evidence[index] = u8::from_str_radix(
            std::str::from_utf8(chunk).unwrap(),
            16,
        )
        .unwrap();
    }

    let admission_source = FixedAdmissionCurrentness(AdmissionContext::active(13, 17));
    let admission = admission_record(&manifest, &manifest_bytes, &component)
        .activate(&manifest, &admission_source)
        .unwrap();
    let worker_path = std::env::var("SYMTHAEA_FILESYSTEM_WORKER_BIN").unwrap();
    let image = SealedWorkerImage::from_path(worker_path).unwrap();
    let qualification_record =
        WorkerQualificationRecord::issue(image.image_sha256(), qualification_evidence, 23).unwrap();
    let worker_source = FixedWorkerCurrentness(WorkerQualificationContext::active(23));
    let qualification = qualification_record.activate(&image, &worker_source).unwrap();
    let deployment = BoundSimulationDeployment::issue(
        &admission,
        &admission_source,
        &qualification,
        &worker_source,
        manifest_bytes,
        component,
        image,
        SupervisorLimits::default(),
        WorkerFrameLimits::default(),
    )
    .unwrap();

    let mut registry = ExtensionRegistry::new();
    registry.register(manifest.clone()).unwrap();
    let observations = vec![observation(&manifest)];
    let authorities = vec![DeploymentExecutionAuthority::new(&deployment, &qualification)];
    let admissions = vec![admission];
    let routed_request = request("real-routed-deployment");
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
        &worker_source,
    )
    .unwrap();
    let completed = selection
        .execute(&routed_request, &admission_source, &worker_source)
        .unwrap();

    assert_eq!(completed.decision().selected, manifest.id);
    assert_eq!(completed.request_sha256(), selection.request_sha256());
    let result = completed.invocation().invocation().invocation().result();
    assert_eq!(result.evidence, SimulationEvidence::default());
    assert_eq!(result.metrics.len(), 1);
    assert_eq!(result.metrics[0].name, "fixture.parameter-sum");
    assert_eq!(result.metrics[0].value, 4.0);
}
