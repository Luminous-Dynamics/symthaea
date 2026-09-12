use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use symthaea_extension_admission::{
    ActiveAdmission, AdmissionContext, AdmissionCurrentnessSource, AdmissionRecord,
    AdmissionSubject, PrincipalId, Sha256Digest, TrustLevel,
};
use symthaea_extension_core::{EffectClass, PermissionSet};
use symthaea_extension_host::ControlHostPolicy;
use symthaea_extension_router::{ProviderObservation, ProviderState, RoutingConstraints};
use symthaea_extension_simulation_backend::WasmSimulationComponentFactory;
use symthaea_sim_bridge::{EngineeringDomain, ExecutionMode, SimulationRequest, SolverKind};
use symthaea_sim_digest::{canonical_output_sha256_v1, canonical_request_sha256_v1};
use symthaea_sim_extension_routing::{
    CurrentnessPhase, LazySimulationError, LazySimulationRegistry,
};

#[derive(Debug, Clone, Copy)]
struct StableCurrentness(AdmissionContext);

impl AdmissionCurrentnessSource for StableCurrentness {
    fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
        Some(self.0)
    }
}

#[derive(Debug)]
struct RevokeOnThirdCheck {
    checks: AtomicUsize,
    generation: u64,
    trust_generation: u64,
}

impl AdmissionCurrentnessSource for RevokeOnThirdCheck {
    fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
        let check = self.checks.fetch_add(1, Ordering::SeqCst);
        if check < 2 {
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
    manifest_sha256: Sha256Digest,
    payload_sha256: Sha256Digest,
    generation: u64,
    trust_generation: u64,
) -> ActiveAdmission {
    let manifest = &factory.descriptor().manifest;
    let source = StableCurrentness(AdmissionContext::active(generation, trust_generation));
    AdmissionRecord::issue(
        manifest.id.clone(),
        manifest.version.clone(),
        manifest_sha256,
        payload_sha256,
        Sha256Digest::new([0x55; 32]),
        PrincipalId::new("local:simulation-integration-authority").unwrap(),
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
        "exercise admitted typed simulation Component",
    )
    .with_parameter("alpha", 2.5, "1", "integration fixture")
    .with_parameter("beta", 1.5, "1", "integration fixture")
}

fn constraints() -> RoutingConstraints {
    RoutingConstraints {
        maximum_effect: EffectClass::Pure,
        minimum_trust: TrustLevel::Trusted,
        ..RoutingConstraints::default()
    }
}

fn hex(bytes: [u8; 32]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in bytes {
        out.push(TABLE[(byte >> 4) as usize] as char);
        out.push(TABLE[(byte & 0x0f) as usize] as char);
    }
    out
}

#[test]
#[ignore = "requires release-built examples/extensions/hello-simulation Component"]
fn selected_admitted_component_executes_and_mints_only_execution_lineage() {
    let factory = load_factory();
    let manifest_sha256 = factory.manifest_digest();
    let payload_sha256 = factory.component_digest();
    let admission = issue_admission(&factory, manifest_sha256, payload_sha256, 7, 77);
    let registry = registry_with(factory);
    let request = request("authorized-component-success");
    let request_sha256 = canonical_request_sha256_v1(&request).unwrap();
    let source = StableCurrentness(AdmissionContext::active(7, 77));

    let (result, decision) = registry
        .run(&request, constraints(), &[admission], &source)
        .unwrap();

    assert_eq!(
        decision.selected.as_str(),
        "org.example.hello-simulation"
    );
    assert_eq!(decision.selected_manifest_sha256, manifest_sha256);
    assert_eq!(decision.selected_payload_sha256, payload_sha256);
    assert_eq!(result.evidence.mode, ExecutionMode::ExtensionComponent);
    assert!(!result.is_engineering_evidence());
    assert_eq!(result.metrics.len(), 1);
    assert_eq!(result.metrics[0].name, "fixture.parameter-sum");
    assert_eq!(result.metrics[0].value, 4.0);
    assert_eq!(result.warnings.len(), 1);
    assert!(result.warnings[0].contains("not engineering evidence"));

    let extension = result
        .evidence
        .extension
        .as_ref()
        .expect("host-owned extension lineage");
    assert_eq!(extension.extension_id, "org.example.hello-simulation");
    assert_eq!(extension.manifest_sha256, hex(*manifest_sha256.as_bytes()));
    assert_eq!(extension.component_sha256, hex(*payload_sha256.as_bytes()));
    assert_eq!(extension.request_sha256, hex(request_sha256));
    assert_eq!(
        extension.output_sha256,
        hex(canonical_output_sha256_v1(&result).unwrap())
    );
}

#[test]
#[ignore = "requires release-built examples/extensions/hello-simulation Component"]
fn wrong_admitted_manifest_digest_fails_before_backend_construction() {
    let factory = load_factory();
    let payload_sha256 = factory.component_digest();
    let admission = issue_admission(
        &factory,
        Sha256Digest::new([0xa5; 32]),
        payload_sha256,
        8,
        88,
    );
    let registry = registry_with(factory);
    let source = StableCurrentness(AdmissionContext::active(8, 88));

    let error = registry
        .run(
            &request("wrong-manifest-binding"),
            constraints(),
            &[admission],
            &source,
        )
        .unwrap_err();

    assert!(matches!(error, LazySimulationError::ManifestDigestMismatch { .. }));
}

#[test]
#[ignore = "requires release-built examples/extensions/hello-simulation Component"]
fn revocation_after_component_execution_withholds_result() {
    let factory = load_factory();
    let admission = issue_admission(
        &factory,
        factory.manifest_digest(),
        factory.component_digest(),
        9,
        99,
    );
    let registry = registry_with(factory);
    let source = RevokeOnThirdCheck {
        checks: AtomicUsize::new(0),
        generation: 9,
        trust_generation: 99,
    };

    let error = registry
        .run(
            &request("revoked-after-component-execution"),
            constraints(),
            &[admission],
            &source,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        LazySimulationError::AdmissionCurrentness {
            phase: CurrentnessPhase::AfterExecution,
            ..
        }
    ));
    assert_eq!(source.checks.load(Ordering::SeqCst), 3);
}
