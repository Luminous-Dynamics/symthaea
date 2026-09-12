use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use symthaea_extension_admission::{
    ActiveAdmission, AdmissionContext, AdmissionCurrentnessSource, AdmissionProblem,
    AdmissionRecord, AdmissionSubject, PrincipalId, Sha256Digest, TrustLevel,
};
use symthaea_extension_core::{EffectClass, PermissionSet, RuntimeKind};
use symthaea_extension_host::ControlHostPolicy;
use symthaea_extension_router::{ProviderObservation, ProviderState, RoutingConstraints};
use symthaea_extension_simulation_backend::WasmSimulationComponentFactory;
use symthaea_sim_bridge::{EngineeringDomain, ExecutionMode, SimulationRequest, SolverKind};
use symthaea_sim_extension_routing::LazySimulationRegistry;
use symthaea_sim_release::{
    SIMULATION_RELEASE_PROFILE_V1, SimulationReleaseError, run_released,
};

#[derive(Debug, Clone, Copy)]
struct StableCurrentness(AdmissionContext);

impl AdmissionCurrentnessSource for StableCurrentness {
    fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
        Some(self.0)
    }
}

#[derive(Debug)]
struct RevokeOnFourthCheck {
    checks: AtomicUsize,
    generation: u64,
    trust_generation: u64,
}

impl AdmissionCurrentnessSource for RevokeOnFourthCheck {
    fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
        let check = self.checks.fetch_add(1, Ordering::SeqCst);
        if check < 3 {
            Some(AdmissionContext::active(
                self.generation,
                self.trust_generation,
            ))
        } else {
            Some(AdmissionContext::revoked(
                self.generation,
                self.trust_generation,
            ))
        }
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
    generation: u64,
    trust_generation: u64,
) -> ActiveAdmission {
    let manifest = &factory.descriptor().manifest;
    let source = StableCurrentness(AdmissionContext::active(generation, trust_generation));
    AdmissionRecord::issue(
        manifest.id.clone(),
        manifest.version.clone(),
        factory.manifest_digest(),
        factory.component_digest(),
        Sha256Digest::new([0x66; 32]),
        PrincipalId::new("local:simulation-release-authority").unwrap(),
        Some(PrincipalId::new("did:example:hello-simulation-publisher").unwrap()),
        TrustLevel::Trusted,
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
    .activate(manifest, &source)
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
        "exercise admitted typed simulation release receipt",
    )
    .with_parameter("alpha", 2.5, "1", "release fixture")
    .with_parameter("beta", 1.5, "1", "release fixture")
}

fn constraints() -> RoutingConstraints {
    RoutingConstraints {
        maximum_effect: EffectClass::Pure,
        minimum_trust: TrustLevel::Trusted,
        ..RoutingConstraints::default()
    }
}

#[test]
#[ignore = "requires release-built examples/extensions/hello-simulation Component"]
fn exact_component_mints_verified_release_receipt_after_fourth_currentness_check() {
    let factory = load_factory();
    let manifest_sha256 = factory.manifest_digest();
    let payload_sha256 = factory.component_digest();
    let admission = issue_admission(&factory, 12, 120);
    let registry = registry_with(factory);
    let request = request("release-receipt-success");
    let source = StableCurrentness(AdmissionContext::active(12, 120));

    let release = run_released(
        &registry,
        &request,
        constraints(),
        &[admission],
        &source,
    )
    .unwrap();

    release.verify(&request).unwrap();
    assert_eq!(
        release.decision().selected.as_str(),
        "org.example.hello-simulation"
    );
    assert_eq!(release.receipt().runtime(), RuntimeKind::Wasm);
    assert_eq!(release.receipt().admission_generation(), 12);
    assert_eq!(release.receipt().trust_generation(), 120);
    assert_eq!(release.receipt().manifest_sha256(), manifest_sha256.0);
    assert_eq!(release.receipt().payload_sha256(), payload_sha256.0);
    assert_eq!(release.result().evidence.mode, ExecutionMode::ExtensionComponent);
    assert!(!release.result().is_engineering_evidence());
    assert_eq!(release.result().metrics.len(), 1);
    assert_eq!(release.result().metrics[0].value, 4.0);

    let persisted = release.evidence();
    assert_eq!(persisted.profile, SIMULATION_RELEASE_PROFILE_V1);
    assert_eq!(persisted.selected_extension, "org.example.hello-simulation");
    assert_eq!(persisted.runtime, RuntimeKind::Wasm);
    assert_eq!(persisted.release_sha256.len(), 64);
    assert_eq!(persisted.evidence_sha256.len(), 64);
}

#[test]
#[ignore = "requires release-built examples/extensions/hello-simulation Component"]
fn revocation_at_receipt_finalization_prevents_receipt_minting() {
    let factory = load_factory();
    let admission = issue_admission(&factory, 13, 130);
    let registry = registry_with(factory);
    let source = RevokeOnFourthCheck {
        checks: AtomicUsize::new(0),
        generation: 13,
        trust_generation: 130,
    };

    let error = run_released(
        &registry,
        &request("revoked-at-release-finalization"),
        constraints(),
        &[admission],
        &source,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        SimulationReleaseError::ReleaseCurrentness(AdmissionProblem::Revoked)
    ));
    assert_eq!(source.checks.load(Ordering::SeqCst), 4);
}
