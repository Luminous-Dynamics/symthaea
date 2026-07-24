// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_fabrication_kernel::{
    AttestationPolicy, CSGNode, ContainmentAction, ExecutionGuard, ExecutionGuardPolicy,
    ExecutionTelemetry, FabricationProcessPolicy, MachineCapabilities, MachineProfile,
    MachineSession, ManifestSignatureVerifier, ManifestSigner, ManufacturingReadyMesh,
    MinimumFeaturePolicy, MockPrinter, PackageInspectionLimits, PrinterApi, ReplayEnvironment,
    SignatureAlgorithm, SliceConfig, ToolpathConfig, Transform3D, ValidatedGCode,
    attest_fabrication_manifest, authorize_print_job, build_fabrication_manifest,
    build_replay_contract, export_3mf_package_with_attestation, inspect_3mf_package,
    negotiate_machine_profile, resolve_to_mesh, sha256, slice_manufacturing_ready,
    submit_authorized_job, try_generate_gcode, verify_attestation_authority,
    verify_attested_3mf_package, verify_replay_contract,
};

struct TestProvider;

impl ManifestSigner for TestProvider {
    fn algorithm(&self) -> SignatureAlgorithm {
        SignatureAlgorithm::Other("test-only-sha256".into())
    }

    fn key_id(&self) -> &str {
        "release-test-key"
    }

    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(sha256(message).0.to_vec())
    }
}

impl ManifestSignatureVerifier for TestProvider {
    fn verify(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String> {
        Ok(
            algorithm == &SignatureAlgorithm::Other("test-only-sha256".into())
                && key_id == "release-test-key"
                && signature == sha256(message).0.as_slice(),
        )
    }
}

#[test]
fn qualified_job_crosses_attested_session_bound_release_lane() {
    let geometry = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
        translate: [10.0, 10.0, 0.5],
        ..Default::default()
    }));
    let ready = ManufacturingReadyMesh::try_new(
        geometry,
        FabricationProcessPolicy::default(),
        MinimumFeaturePolicy::default(),
    )
    .unwrap();
    let slice_config = SliceConfig::default();
    let toolpath_config = ToolpathConfig::default();
    let profile = MachineProfile::default();
    let layers = slice_manufacturing_ready(&ready, &slice_config).unwrap();
    let program = try_generate_gcode(&layers, &slice_config, &toolpath_config).unwrap();
    let validated = ValidatedGCode::try_new(program, &profile).unwrap();
    let manifest = build_fabrication_manifest(
        &ready,
        &slice_config,
        &toolpath_config,
        &profile,
        &layers,
        &validated,
    )
    .unwrap();

    let provider = TestProvider;
    let attested = attest_fabrication_manifest(manifest.clone(), &[&provider]).unwrap();
    let package = export_3mf_package_with_attestation(ready.mesh(), &attested).unwrap();
    let inspected = inspect_3mf_package(&package, PackageInspectionLimits::default()).unwrap();
    assert_eq!(inspected.manifest.as_ref(), Some(&manifest));
    assert!(
        verify_attested_3mf_package(
            &package,
            PackageInspectionLimits::default(),
            &AttestationPolicy::default(),
            &provider,
        )
        .unwrap()
        .trusted()
    );

    let environment = ReplayEnvironment {
        kernel_version: env!("CARGO_PKG_VERSION").into(),
        source_revision: "release-test-revision".into(),
        target_triple: "test-target".into(),
        rustc_version: "test-rustc".into(),
        cargo_lock_digest: None,
        feature_flags: vec!["analytical".into()],
    };
    let replay = build_replay_contract(&manifest, environment.clone(), 0).unwrap();
    assert!(
        verify_replay_contract(&replay, &manifest, environment)
            .unwrap()
            .reproducible()
    );

    let verified =
        verify_attestation_authority(attested, &AttestationPolicy::default(), &provider).unwrap();
    let negotiated = negotiate_machine_profile(
        &profile,
        MachineSession {
            session_nonce: "release-session-1".into(),
            capabilities: MachineCapabilities::from_profile("fabricator-01", &profile),
        },
    )
    .unwrap();
    let authorized = authorize_print_job(validated, verified, negotiated).unwrap();
    let mut printer = MockPrinter::new();
    printer.connect().unwrap();
    let receipt = submit_authorized_job(
        &mut printer,
        authorized,
        "fabricator-01",
        "release-session-1",
    )
    .unwrap();
    assert_eq!(receipt.machine_id, "fabricator-01");

    let mut guard = ExecutionGuard::new(ExecutionGuardPolicy::default()).unwrap();
    let healthy = guard.observe(ExecutionTelemetry {
        elapsed_s: 0.0,
        heartbeat_sequence: 1,
        progress: 0.0,
        nozzle_actual_c: 200.0,
        nozzle_target_c: 200.0,
        bed_actual_c: 60.0,
        bed_target_c: 60.0,
    });
    assert_eq!(healthy.action, ContainmentAction::Continue);
    let emergency = guard.observe(ExecutionTelemetry {
        elapsed_s: 1.0,
        heartbeat_sequence: 2,
        progress: 0.01,
        nozzle_actual_c: 321.0,
        nozzle_target_c: 200.0,
        bed_actual_c: 60.0,
        bed_target_c: 60.0,
    });
    assert_eq!(emergency.latched_action, ContainmentAction::EmergencyStop);
}
