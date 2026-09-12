use symthaea_extension_admission::{
    AdmissionContext, AdmissionCurrentnessSource, AdmissionRecord, AdmissionSubject, PrincipalId,
    Sha256Digest, TrustLevel,
};
use symthaea_extension_core::{
    AbiVersion, CapabilityDescriptor, EffectClass, ExtensionId, ExtensionKind, ExtensionManifest,
    PermissionSet, ResourceBudget, RuntimeKind,
};
use symthaea_extension_router::{ProviderObservation, ProviderState, RoutingConstraints};
use symthaea_sim_bridge::{
    EngineeringDomain, ExecutionMode, ExtensionComponentEvidence, SimulationBackend,
    SimulationError, SimulationEvidence, SimulationRequest, SimulationResult, SolverKind,
};
use symthaea_sim_digest::{
    SIMULATION_DIGEST_PROFILE_V1, canonical_output_sha256_v1, canonical_request_sha256_v1,
};
use symthaea_sim_extension_routing::{
    SelectedExecutionPermit, SimulationBackendFactory, SimulationProviderDescriptor,
    LazySimulationRegistry, solver_capability,
};
use symthaea_sim_release::{SimulationReleaseError, run_released};

#[derive(Debug, Clone, Copy)]
struct StableCurrentness(AdmissionContext);

impl AdmissionCurrentnessSource for StableCurrentness {
    fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
        Some(self.0)
    }
}

#[derive(Debug)]
struct LyingWasmFactory {
    descriptor: SimulationProviderDescriptor,
    admitted_manifest: Sha256Digest,
    admitted_payload: Sha256Digest,
}

impl LyingWasmFactory {
    fn new(admitted_manifest: Sha256Digest, admitted_payload: Sha256Digest) -> Self {
        let capability = solver_capability(SolverKind::Custom);
        Self {
            descriptor: SimulationProviderDescriptor {
                manifest: ExtensionManifest {
                    id: ExtensionId::new("org.example.liar"),
                    name: "lineage substitution fixture".into(),
                    version: "1.0.0".into(),
                    abi: AbiVersion::V1,
                    kind: ExtensionKind::Simulation,
                    runtime: RuntimeKind::Wasm,
                    description: "host-side lying factory fixture".into(),
                    provides: vec![CapabilityDescriptor {
                        id: capability,
                        description: "synthetic custom simulation".into(),
                        effect: EffectClass::Pure,
                    }],
                    requires: vec![],
                    permissions: PermissionSet::default(),
                    resources: ResourceBudget::default(),
                },
                backend_name: "lying-extension-component".into(),
                supported_solvers: vec![SolverKind::Custom],
            },
            admitted_manifest,
            admitted_payload,
        }
    }
}

impl SimulationBackendFactory for LyingWasmFactory {
    fn descriptor(&self) -> &SimulationProviderDescriptor {
        &self.descriptor
    }

    fn manifest_sha256(&self) -> Option<Sha256Digest> {
        Some(self.admitted_manifest)
    }

    fn executable_payload_sha256(&self) -> Option<Sha256Digest> {
        Some(self.admitted_payload)
    }

    fn create(
        &self,
        _permit: &SelectedExecutionPermit,
    ) -> Result<Box<dyn SimulationBackend>, SimulationError> {
        Ok(Box::new(LyingBackend))
    }
}

#[derive(Debug)]
struct LyingBackend;

impl SimulationBackend for LyingBackend {
    fn name(&self) -> &'static str {
        "lying-extension-component"
    }

    fn supported_solvers(&self) -> &[SolverKind] {
        &[SolverKind::Custom]
    }

    fn run(&self, request: &SimulationRequest) -> Result<SimulationResult, SimulationError> {
        let mut result = SimulationResult::converged(request.id.clone(), 0.0)
            .with_metric("fixture.value", 1.0, "1");
        let request_sha256 = canonical_request_sha256_v1(request).unwrap();
        let output_sha256 = canonical_output_sha256_v1(&result).unwrap();
        result.evidence = SimulationEvidence {
            mode: ExecutionMode::ExtensionComponent,
            backend: Some("lying-extension-component".into()),
            extension: Some(ExtensionComponentEvidence {
                extension_id: "org.example.liar".into(),
                extension_version: "1.0.0".into(),
                // Canonical-looking but deliberately unrelated to the factory's
                // admission-matching digest claims.
                manifest_sha256: "aa".repeat(32),
                component_sha256: "bb".repeat(32),
                runtime_profile: "lying-runtime".into(),
                digest_profile: SIMULATION_DIGEST_PROFILE_V1.into(),
                request_sha256: hex(request_sha256),
                output_sha256: hex(output_sha256),
                wit_version: "simulation-provider-v1".into(),
                adapter_version: "lying-adapter".into(),
            }),
            ..SimulationEvidence::default()
        };
        Ok(result)
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
fn release_rejects_factory_claims_that_disagree_with_execution_lineage() {
    let manifest_sha256 = Sha256Digest::new([0x10; 32]);
    let payload_sha256 = Sha256Digest::new([0x11; 32]);
    let factory = LyingWasmFactory::new(manifest_sha256, payload_sha256);
    let manifest = factory.descriptor().manifest.clone();
    let currentness = StableCurrentness(AdmissionContext::active(4, 44));
    let admission = AdmissionRecord::issue(
        manifest.id.clone(),
        manifest.version.clone(),
        manifest_sha256,
        payload_sha256,
        Sha256Digest::new([0x12; 32]),
        PrincipalId::new("local:lineage-test-authority").unwrap(),
        Some(PrincipalId::new("did:example:lineage-test-signer").unwrap()),
        TrustLevel::Trusted,
        manifest
            .provides
            .iter()
            .map(|capability| capability.id.clone())
            .collect(),
        PermissionSet::default(),
        4,
        44,
    )
    .unwrap()
    .activate(&manifest, &currentness)
    .unwrap();

    let mut registry = LazySimulationRegistry::new();
    registry.register(factory).unwrap();
    registry
        .set_observation(ProviderObservation {
            extension: manifest.id,
            state: ProviderState::Ready,
            evidence_grade: 1,
            reliability_bps: 10_000,
            estimated_latency_ms: Some(1),
        })
        .unwrap();

    let request = SimulationRequest::new(
        "lineage-substitution",
        EngineeringDomain::Systems,
        SolverKind::Custom,
        "prove release checks execution lineage rather than factory digest claims",
    );
    let error = run_released(
        &registry,
        &request,
        RoutingConstraints {
            maximum_effect: EffectClass::Pure,
            minimum_trust: TrustLevel::Trusted,
            ..RoutingConstraints::default()
        },
        &[admission],
        &currentness,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        SimulationReleaseError::ExtensionLineageMismatch("manifest digest")
    ));
}
