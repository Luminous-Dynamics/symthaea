// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;
use symthaea_fabrication_kernel::{
    AttestationPolicy, AuditAction, CSGNode, ContainmentAction, ExecutionGuard,
    ExecutionGuardPolicy, ExecutionTelemetry, FabricationGovernance, FabricationProcessPolicy,
    InterruptedPrintEvidence, KeyLifecycleStatus, KeyTrustRecord, KeyUsage, MachineCapabilities,
    MachineProfile, MachineSession, ManifestSignatureVerifier, ManifestSigner,
    ManufacturingReadyMesh, MinimumFeaturePolicy, MockPrinter, PackageInspectionLimits, PrinterApi,
    RecoveryPolicy, ReplayEnvironment, SignatureAlgorithm, SliceConfig, ToolpathConfig,
    Transform3D, TrustSnapshot, ValidatedGCode, attest_fabrication_manifest,
    build_fabrication_manifest, build_governed_replay_contract, export_3mf_package_with_governance,
    negotiate_machine_profile, reauthorize_print_restart, resolve_to_mesh, sha256,
    slice_manufacturing_ready, submit_authorized_job, try_generate_gcode,
    verify_governed_3mf_package, verify_governed_replay_contract,
};

struct TestProvider;

impl ManifestSigner for TestProvider {
    fn algorithm(&self) -> SignatureAlgorithm {
        SignatureAlgorithm::Other("governed-test-sha256".into())
    }

    fn key_id(&self) -> &str {
        "governed-release-key"
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
            algorithm == &SignatureAlgorithm::Other("governed-test-sha256".into())
                && key_id == "governed-release-key"
                && signature == sha256(message).0.as_slice(),
        )
    }
}

#[test]
fn governed_release_binds_lifecycle_audit_replay_and_recovery() {
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
    let trust = TrustSnapshot::new(
        42,
        100,
        1_000,
        vec![KeyTrustRecord {
            algorithm: SignatureAlgorithm::Other("governed-test-sha256".into()),
            key_id: "governed-release-key".into(),
            not_before_unix_s: 100,
            not_after_unix_s: Some(900),
            status: KeyLifecycleStatus::Active,
            usages: BTreeSet::from([KeyUsage::FabricationManifest]),
        }],
    )
    .unwrap();
    let mut governance = FabricationGovernance::new("release-operator", 500, trust).unwrap();
    let verified = governance
        .verify_attestation(
            500,
            attested.clone(),
            &AttestationPolicy::default(),
            &provider,
        )
        .unwrap();

    let first_machine = negotiate_machine_profile(
        &profile,
        MachineSession {
            session_nonce: "session-1".into(),
            capabilities: MachineCapabilities::from_profile("fabricator-01", &profile),
        },
    )
    .unwrap();
    let authorized = governance
        .authorize_job(501, validated.clone(), verified.clone(), first_machine)
        .unwrap();
    let mut printer = MockPrinter::new();
    printer.connect().unwrap();
    let receipt =
        submit_authorized_job(&mut printer, authorized, "fabricator-01", "session-1").unwrap();
    governance.record_submission(502, &receipt).unwrap();

    let mut guard_policy = ExecutionGuardPolicy::default();
    guard_policy.progress_stall_timeout_s = 5.0;
    let mut guard = ExecutionGuard::new(guard_policy).unwrap();
    guard.observe(ExecutionTelemetry {
        elapsed_s: 0.0,
        heartbeat_sequence: 1,
        progress: 0.2,
        nozzle_actual_c: 200.0,
        nozzle_target_c: 200.0,
        bed_actual_c: 60.0,
        bed_target_c: 60.0,
    });
    let pause = guard.observe(ExecutionTelemetry {
        elapsed_s: 6.0,
        heartbeat_sequence: 2,
        progress: 0.2,
        nozzle_actual_c: 200.0,
        nozzle_target_c: 200.0,
        bed_actual_c: 60.0,
        bed_target_c: 60.0,
    });
    assert_eq!(pause.latched_action, ContainmentAction::Pause);
    governance
        .record_containment(503, manifest_digest(&receipt), &pause)
        .unwrap();

    let evidence = InterruptedPrintEvidence::new(receipt.clone(), guard.checkpoint(), 503).unwrap();
    let recovery_machine = negotiate_machine_profile(
        &profile,
        MachineSession {
            session_nonce: "session-2".into(),
            capabilities: MachineCapabilities::from_profile("fabricator-01", &profile),
        },
    )
    .unwrap();
    let recovered = reauthorize_print_restart(
        validated.clone(),
        verified,
        recovery_machine,
        &evidence,
        510,
        RecoveryPolicy::default(),
    )
    .unwrap();
    assert_eq!(recovered.session_nonce(), "session-2");

    let package = export_3mf_package_with_governance(
        ready.mesh(),
        &attested,
        governance.trust_snapshot(),
        governance.audit_journal(),
    )
    .unwrap();
    assert!(
        verify_governed_3mf_package(
            &package,
            PackageInspectionLimits::default(),
            &AttestationPolicy::default(),
            &provider,
            500,
        )
        .unwrap()
        .trusted()
    );

    let environment = ReplayEnvironment {
        kernel_version: env!("CARGO_PKG_VERSION").into(),
        source_revision: "governed-release-test".into(),
        target_triple: "test-target".into(),
        rustc_version: "test-rustc".into(),
        cargo_lock_digest: None,
        feature_flags: vec!["analytical".into()],
    };
    let replay = build_governed_replay_contract(
        &manifest,
        environment.clone(),
        0,
        governance.trust_snapshot(),
        governance.audit_journal(),
    )
    .unwrap();
    assert!(
        verify_governed_replay_contract(
            &replay,
            &manifest,
            environment,
            governance.trust_snapshot(),
            governance.audit_journal(),
        )
        .unwrap()
        .reproducible()
    );
    assert!(
        governance
            .audit_journal()
            .events
            .iter()
            .any(|event| event.action == AuditAction::ExecutionPaused)
    );
}

fn manifest_digest(
    receipt: &symthaea_fabrication_kernel::SubmittedJobReceipt,
) -> symthaea_fabrication_kernel::Sha256Digest {
    receipt.manifest_digest
}
