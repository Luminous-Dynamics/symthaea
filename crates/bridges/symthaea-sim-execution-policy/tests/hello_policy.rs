use std::fs;
use std::path::PathBuf;
use symthaea_extension_admission::{
    ActiveAdmission, AdmissionContext, AdmissionCurrentnessSource, AdmissionRecord,
    AdmissionSubject, PrincipalId, Sha256Digest, TrustLevel,
};
use symthaea_extension_core::{EffectClass, PermissionSet, RuntimeKind};
use symthaea_extension_host::ControlHostPolicy;
use symthaea_extension_router::{ProviderObservation, ProviderState, RoutingConstraints};
use symthaea_extension_simulation_backend::WasmSimulationComponentFactory;
use symthaea_sim_bridge::{EngineeringDomain, SimulationRequest, SolverKind};
use symthaea_sim_execution_policy::{
    SIMULATION_EXECUTION_POLICY_PROFILE_V1, SimulationExecutionPolicy,
    SimulationExecutionPolicyError, WasmCompilationPolicy, run_with_execution_policy,
};
use symthaea_sim_extension_routing::{LazySimulationError, LazySimulationRegistry};
use symthaea_sim_release::SimulationReleaseError;

#[derive(Debug, Clone, Copy)]
struct StableCurrentness(AdmissionContext);

impl AdmissionCurrentnessSource for StableCurrentness {
    fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
        Some(self.0)
    }
}

#[derive(Debug)]
struct PanicCurrentness;

impl AdmissionCurrentnessSource for PanicCurrentness {
    fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
        panic!("execution policy should reject before a live currentness check")
    }
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../examples/extensions/hello-simulation")
}

fn load_factory() -> WasmSimulationComponentFactory {
    let fixture = fixture_dir();
    let manifest = fs::read(fixture.join("manifest.json")).expect("read fixture manifest");
    let component = fs::read(
        fixture.join("target/wasm32-wasip2/release/symthaea_hello_simulation_extension.wasm"),
    )
    .expect("read release-built hello-simulation Component");
    WasmSimulationComponentFactory::new(
        manifest,
        component,
        vec![SolverKind::Custom],
        ControlHostPolicy::default(),
    )
    .expect("construct exact-byte Wasm simulation factory")
}

fn issue_admission(
    factory: &WasmSimulationComponentFactory,
    trust: TrustLevel,
    generation: u64,
    trust_generation: u64,
) -> ActiveAdmission {
    let manifest = &factory.descriptor().manifest;
    let activation = StableCurrentness(AdmissionContext::active(generation, trust_generation));
    AdmissionRecord::issue(
        manifest.id.clone(),
        manifest.version.clone(),
        factory.manifest_digest(),
        factory.component_digest(),
        Sha256Digest::new([0x77; 32]),
        PrincipalId::new("local:execution-policy-authority").unwrap(),
        Some(PrincipalId::new("did:example:hello-simulation-publisher").unwrap()),
        trust,
        manifest
            .provides
            .iter()
            .map(|capability| capability.id.clone())
            .collect(),
        PermissionSet::default(),
        generation,
        trust_generation,
    )
    .unwrap()
    .activate(manifest, &activation)
    .unwrap()
}

fn registry_with(factory: WasmSimulationComponentFactory) -> LazySimulationRegistry {
    let extension = factory.descriptor().manifest.id.clone();
    let mut registry = LazySimulationRegistry::new();
    registry.register(factory).unwrap();
    registry
        .set_observation(ProviderObservation {
            extension,
            state: ProviderState::Ready,
            evidence_grade: 1,
            reliability_bps: 10_000,
            estimated_latency_ms: Some(10),
        })
        .unwrap();
    registry
}

fn request(id: &str) -> SimulationRequest {
    SimulationRequest::new(
        id,
        EngineeringDomain::Systems,
        SolverKind::Custom,
        "exercise execution-substrate policy",
    )
    .with_parameter("alpha", 2.5, "1", "policy fixture")
    .with_parameter("beta", 1.5, "1", "policy fixture")
}

fn constraints() -> RoutingConstraints {
    RoutingConstraints {
        maximum_effect: EffectClass::Pure,
        ..RoutingConstraints::default()
    }
}

#[test]
#[ignore = "requires release-built examples/extensions/hello-simulation Component"]
fn default_policy_rejects_in_process_wasm_before_currentness_or_execution() {
    let factory = load_factory();
    let admission = issue_admission(&factory, TrustLevel::Trusted, 21, 210);
    let registry = registry_with(factory);

    let error = run_with_execution_policy(
        &registry,
        &request("default-rejects-wasm"),
        constraints(),
        &[admission],
        &PanicCurrentness,
        SimulationExecutionPolicy::default(),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        SimulationExecutionPolicyError::Release(SimulationReleaseError::Execution(
            LazySimulationError::Route(_)
        ))
    ));
}

#[test]
#[ignore = "requires release-built examples/extensions/hello-simulation Component"]
fn explicit_trusted_in_process_exception_is_bound_to_successful_release() {
    let factory = load_factory();
    let admission = issue_admission(&factory, TrustLevel::Trusted, 22, 220);
    let registry = registry_with(factory);
    let source = StableCurrentness(AdmissionContext::active(22, 220));
    let request = request("trusted-in-process-opt-in");

    let release = run_with_execution_policy(
        &registry,
        &request,
        constraints(),
        &[admission],
        &source,
        SimulationExecutionPolicy::allow_trusted_in_process_wasm(),
    )
    .unwrap();

    release.verify(&request).unwrap();
    assert_eq!(release.release().receipt().runtime(), RuntimeKind::Wasm);
    assert_eq!(release.result().metrics.len(), 1);
    assert_eq!(release.result().metrics[0].value, 4.0);

    let evidence = release.evidence();
    assert_eq!(evidence.profile, SIMULATION_EXECUTION_POLICY_PROFILE_V1);
    assert_eq!(
        evidence.wasm_compilation,
        WasmCompilationPolicy::AllowTrustedInProcess
    );
    assert_eq!(evidence.effective_minimum_trust, TrustLevel::Trusted);
    assert_eq!(evidence.selected_trust, TrustLevel::Trusted);
    assert!(evidence
        .effective_allowed_runtimes
        .contains(&RuntimeKind::Wasm));
    assert_eq!(evidence.policy_release_sha256.len(), 64);
    assert_eq!(
        evidence.underlying_release_sha256,
        evidence.release.release_sha256
    );
}

#[test]
#[ignore = "requires release-built examples/extensions/hello-simulation Component"]
fn in_process_exception_still_rejects_community_admission_before_execution() {
    let factory = load_factory();
    let admission = issue_admission(&factory, TrustLevel::Community, 23, 230);
    let registry = registry_with(factory);

    let error = run_with_execution_policy(
        &registry,
        &request("community-still-rejected"),
        constraints(),
        &[admission],
        &PanicCurrentness,
        SimulationExecutionPolicy::allow_trusted_in_process_wasm(),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        SimulationExecutionPolicyError::Release(SimulationReleaseError::Execution(
            LazySimulationError::Route(_)
        ))
    ));
}
