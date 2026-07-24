// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;
use symthaea_fabrication_kernel::{
    AttestationPolicy, AuditAnchorSigner, AuditAnchorVerifier, CSGNode, ExecutionGuardPolicy,
    FabricationGovernance, FabricationProcessPolicy, KeyLifecycleStatus, KeyTrustRecord, KeyUsage,
    MachineCapabilities, MachineProfile, MachineSession, ManifestSignatureVerifier, ManifestSigner,
    ManufacturingReadyMesh, MinimumFeaturePolicy, MockPrinter, PackageInspectionLimits, PrinterApi,
    ReleasePolicy, ReleaseQuorumRequirement, ReleaseSignerBinding, ReplayEnvironment,
    SignatureAlgorithm, SignerRole, SliceConfig, TimedMachineSession, ToolpathConfig, Transform3D,
    TrustSnapshot, ValidatedGCode, attest_fabrication_manifest, build_fabrication_manifest,
    build_governed_replay_contract, build_operational_replay_contract, export_audit_segment,
    negotiate_machine_profile_at, resolve_to_mesh, run_standard_fault_matrix, sha256,
    sign_audit_anchor, slice_manufacturing_ready, submit_governed_authorized_job,
    try_generate_gcode, verify_audit_segment, verify_operational_replay_contract,
    verify_signed_audit_anchor,
};

struct ManifestProvider {
    key_id: &'static str,
    algorithm: SignatureAlgorithm,
}

impl ManifestSigner for ManifestProvider {
    fn algorithm(&self) -> SignatureAlgorithm {
        self.algorithm.clone()
    }
    fn key_id(&self) -> &str {
        self.key_id
    }
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(signature(self.key_id, message))
    }
}

struct Verifier;
impl ManifestSignatureVerifier for Verifier {
    fn verify(
        &self,
        _algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature_bytes: &[u8],
    ) -> Result<bool, String> {
        Ok(signature_bytes == signature(key_id, message))
    }
}

struct AuditProvider;
impl AuditAnchorSigner for AuditProvider {
    fn algorithm(&self) -> SignatureAlgorithm {
        SignatureAlgorithm::Ed25519
    }
    fn key_id(&self) -> &str {
        "audit-root"
    }
    fn sign_audit_anchor(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(signature(self.key_id(), message))
    }
}
impl AuditAnchorVerifier for AuditProvider {
    fn verify_audit_anchor(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature_bytes: &[u8],
    ) -> Result<bool, String> {
        Ok(algorithm == &SignatureAlgorithm::Ed25519
            && key_id == self.key_id()
            && signature_bytes == signature(key_id, message))
    }
}

#[test]
fn operational_release_binds_quorum_session_audit_faults_and_replay() {
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

    let design = ManifestProvider {
        key_id: "design-root",
        algorithm: SignatureAlgorithm::Ed25519,
    };
    let safety = ManifestProvider {
        key_id: "safety-root",
        algorithm: SignatureAlgorithm::MlDsa65,
    };
    let attested = attest_fabrication_manifest(manifest.clone(), &[&design, &safety]).unwrap();
    let trust = TrustSnapshot::new(
        7,
        100,
        1_000,
        vec![
            trust_key(
                SignatureAlgorithm::Ed25519,
                "design-root",
                BTreeSet::from([KeyUsage::FabricationManifest]),
            ),
            trust_key(
                SignatureAlgorithm::MlDsa65,
                "safety-root",
                BTreeSet::from([KeyUsage::FabricationManifest]),
            ),
            trust_key(
                SignatureAlgorithm::Ed25519,
                "audit-root",
                BTreeSet::from([KeyUsage::AuditAnchor]),
            ),
        ],
    )
    .unwrap();
    let mut governance = FabricationGovernance::new("release-operator", 500, trust).unwrap();
    let verified = governance
        .verify_attestation(
            500,
            attested,
            &AttestationPolicy {
                minimum_valid_signatures: 2,
                ..Default::default()
            },
            &Verifier,
        )
        .unwrap();

    let release_policy = ReleasePolicy::new(
        2,
        4,
        true,
        vec![
            ReleaseQuorumRequirement {
                role: SignerRole::DesignAuthority,
                minimum_distinct_signers: 1,
            },
            ReleaseQuorumRequirement {
                role: SignerRole::SafetyAuthority,
                minimum_distinct_signers: 1,
            },
        ],
        vec![
            ReleaseSignerBinding {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "design-root".into(),
                roles: BTreeSet::from([SignerRole::DesignAuthority]),
            },
            ReleaseSignerBinding {
                algorithm: SignatureAlgorithm::MlDsa65,
                key_id: "safety-root".into(),
                roles: BTreeSet::from([SignerRole::SafetyAuthority]),
            },
        ],
    )
    .unwrap();
    let release = governance
        .authorize_release(501, &verified, &release_policy)
        .unwrap();

    let timed_session = TimedMachineSession::new(
        MachineSession {
            session_nonce: "session-7".into(),
            capabilities: MachineCapabilities::from_profile("fabricator-01", &profile),
        },
        7,
        500,
        600,
    );
    governance
        .accept_machine_session(502, &timed_session)
        .unwrap();
    let machine = negotiate_machine_profile_at(&profile, timed_session, 502).unwrap();
    let authorized = governance
        .authorize_governed_job(503, validated, verified, &release, machine)
        .unwrap();
    let mut printer = MockPrinter::new();
    printer.connect().unwrap();
    let receipt =
        submit_governed_authorized_job(&mut printer, authorized, "fabricator-01", "session-7", 504)
            .unwrap();
    governance
        .record_governed_submission(505, &receipt)
        .unwrap();

    let fault_reports = run_standard_fault_matrix(ExecutionGuardPolicy::default()).unwrap();
    governance
        .record_fault_injection_matrix(506, manifest_digest(&receipt), &fault_reports)
        .unwrap();

    let signed_anchor = sign_audit_anchor(
        governance.audit_journal(),
        governance.trust_snapshot(),
        507,
        "release-anchor-7",
        &AuditProvider,
    )
    .unwrap();
    let verified_anchor = verify_signed_audit_anchor(
        &signed_anchor,
        governance.audit_journal(),
        governance.trust_snapshot(),
        507,
        &AuditProvider,
    )
    .unwrap();

    let environment = ReplayEnvironment {
        kernel_version: env!("CARGO_PKG_VERSION").into(),
        source_revision: "operational-release-test".into(),
        target_triple: "test-target".into(),
        rustc_version: "test-rustc".into(),
        cargo_lock_digest: None,
        feature_flags: vec!["analytical".into()],
    };
    let governed_replay = build_governed_replay_contract(
        &manifest,
        environment,
        0,
        governance.trust_snapshot(),
        governance.audit_journal(),
    )
    .unwrap();
    let operational = build_operational_replay_contract(
        governed_replay.clone(),
        &release,
        &receipt,
        &verified_anchor,
        &fault_reports,
    )
    .unwrap();
    assert!(
        verify_operational_replay_contract(
            &operational,
            governed_replay,
            &release,
            &receipt,
            &verified_anchor,
            &fault_reports,
        )
        .unwrap()
        .reproducible()
    );

    let segment = export_audit_segment(governance.audit_journal(), 2, 3).unwrap();
    assert!(
        verify_audit_segment(
            &segment,
            Some(governance.audit_journal().events[0].record_hash),
        )
        .intact()
    );
    assert!(
        governance
            .session_tracker()
            .contains_consumed_nonce("fabricator-01", "session-7")
    );

    let _inspection_limits = PackageInspectionLimits::default();
}

fn trust_key(
    algorithm: SignatureAlgorithm,
    key_id: &str,
    usages: BTreeSet<KeyUsage>,
) -> KeyTrustRecord {
    KeyTrustRecord {
        algorithm,
        key_id: key_id.into(),
        not_before_unix_s: 1,
        not_after_unix_s: None,
        status: KeyLifecycleStatus::Active,
        usages,
    }
}

fn signature(key_id: &str, message: &[u8]) -> Vec<u8> {
    let mut bytes = key_id.as_bytes().to_vec();
    bytes.extend_from_slice(message);
    sha256(&bytes).0.to_vec()
}

fn manifest_digest(
    receipt: &symthaea_fabrication_kernel::GovernedSubmittedJobReceipt,
) -> symthaea_fabrication_kernel::Sha256Digest {
    receipt.submission.manifest_digest
}
