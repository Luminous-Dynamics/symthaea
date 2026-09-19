// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;

use crate::*;

struct Signer { key_id: &'static str }
impl AttestationSigner for Signer {
    fn algorithm(&self) -> SignatureAlgorithm { SignatureAlgorithm::Ed25519 }
    fn key_id(&self) -> &str { self.key_id }
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(Sha256Digest::of_bytes(message).as_str().as_bytes().to_vec())
    }
}

struct Verifier;
impl AttestationSignatureVerifier for Verifier {
    fn verify(
        &self,
        _algorithm: &SignatureAlgorithm,
        key_id: &str,
        expected_verification_key_sha256: Option<&Sha256Digest>,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String> {
        let resolved = key_material(key_id);
        if expected_verification_key_sha256.is_some_and(|expected| expected != &resolved) {
            return Err("verification-key digest mismatch".into());
        }
        Ok(signature == Sha256Digest::of_bytes(message).as_str().as_bytes())
    }
}

struct Bootstrap;
impl GenesisTrustAnchorVerifier for Bootstrap {
    fn verifier_authority_sha256(&self) -> Sha256Digest {
        Sha256Digest::of_bytes(b"trusted-time-test-bootstrap")
    }
    fn verify_genesis_state(
        &self,
        _root_sha256: &Sha256Digest,
        _principal_directory_sha256: &Sha256Digest,
        _initial_trust_snapshot_sha256: &Sha256Digest,
        _anchor_artifact_sha256: &Sha256Digest,
        _anchored_at_unix_s: u64,
    ) -> Result<bool, String> { Ok(true) }
}

fn key_material(key_id: &str) -> Sha256Digest {
    Sha256Digest::of_bytes(format!("verification-key:{key_id}").as_bytes())
}
fn usage(role: TrustRole) -> TrustUsage {
    let suffix = match role {
        TrustRole::Root => "root",
        TrustRole::Freshness => "freshness",
        TrustRole::KeyLifecycle => "key-lifecycle",
        TrustRole::QualificationProfile => "qualification-profile",
        TrustRole::QualificationDecision => "qualification-decision",
        TrustRole::QualificationLifecycle => "qualification-lifecycle",
        TrustRole::TransparencyLog => "transparency-log",
        TrustRole::TransparencyWitness => "transparency-witness",
        TrustRole::EmergencyRecovery => "emergency-recovery",
    };
    TrustUsage::parse(format!("trust.role.{suffix}")).unwrap()
}
fn principal(id: &str, org: &str, region: &str, key_id: &str, roles: &[TrustRole]) -> TrustedPrincipal {
    TrustedPrincipal {
        principal_id: id.into(), organization_id: org.into(), region_id: region.into(),
        keys: vec![TrustedKeyBinding {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: key_id.into(),
            verification_key_sha256: key_material(key_id),
            roles: roles.iter().copied().collect(),
        }],
    }
}
fn role_policy(role: TrustRole) -> TrustRolePolicy {
    TrustRolePolicy {
        role,
        minimum_valid_signatures: 1,
        minimum_distinct_principals: 1,
        minimum_distinct_organizations: 1,
        minimum_distinct_regions: 1,
        required_algorithms: BTreeSet::from([SignatureAlgorithm::Ed25519]),
        allowed_principal_ids: None,
    }
}
fn fixture() -> (FrozenTrustRoot, TrustedPrincipalDirectory, TrustSnapshot, AuthorizedTrustState) {
    let directory = TrustedPrincipalDirectory::new(1, 100, vec![
        principal("root", "org-root", "region-root", "root-key", &[TrustRole::Root, TrustRole::Freshness]),
        principal("log", "org-log", "region-log", "log-key", &[TrustRole::TransparencyLog]),
        principal("witness", "org-witness", "region-witness", "witness-key", &[TrustRole::TransparencyWitness]),
    ]).unwrap();
    let root = TrustRootDraft {
        version: 1,
        predecessor_root_sha256: None,
        issued_at_unix_s: 100,
        expires_at_unix_s: 900,
        role_policies: vec![
            role_policy(TrustRole::Root),
            role_policy(TrustRole::Freshness),
            role_policy(TrustRole::TransparencyLog),
            role_policy(TrustRole::TransparencyWitness),
        ],
    }.freeze(&directory).unwrap();
    let snapshot = TrustSnapshot::new(1, 100, 900, vec![
        KeyTrustRecord {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "root-key".into(), verification_key_sha256: key_material("root-key"),
            not_before_unix_s: 100, not_after_unix_s: Some(900), status: KeyLifecycleStatus::Active,
            usages: BTreeSet::from([usage(TrustRole::Root), usage(TrustRole::Freshness)]),
        },
        KeyTrustRecord {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "log-key".into(), verification_key_sha256: key_material("log-key"),
            not_before_unix_s: 100, not_after_unix_s: Some(900), status: KeyLifecycleStatus::Active,
            usages: BTreeSet::from([usage(TrustRole::TransparencyLog)]),
        },
        KeyTrustRecord {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "witness-key".into(), verification_key_sha256: key_material("witness-key"),
            not_before_unix_s: 100, not_after_unix_s: Some(900), status: KeyLifecycleStatus::Active,
            usages: BTreeSet::from([usage(TrustRole::TransparencyWitness)]),
        },
    ]).unwrap();
    let state = authorize_genesis_trust_state(
        &root, &directory, &snapshot,
        &GenesisTrustAnchorEvidence {
            anchor_artifact_sha256: Sha256Digest::of_bytes(b"offline-root-anchor"),
            initial_trust_snapshot_sha256: snapshot.digest().unwrap(),
            anchored_at_unix_s: 101,
        },
        &Bootstrap,
    ).unwrap();
    (root, directory, snapshot, state)
}

fn authorize_statement(
    root: &FrozenTrustRoot,
    directory: &TrustedPrincipalDirectory,
    snapshot: &TrustSnapshot,
    state: &AuthorizedTrustState,
    role: TrustRole,
    key_id: &'static str,
    statement: &TimeSourceStatement,
) -> AuthorizedTrustRoleAttestation {
    let signer = Signer { key_id };
    let envelope = attest_digests(
        usage(role),
        statement.source_artifact_sha256().clone(),
        statement.statement_sha256().clone(),
        Some(root.root_sha256().clone()),
        &[&signer],
    ).unwrap();
    let mut tracker = TrustSnapshotTracker::default();
    tracker.accept(snapshot).unwrap();
    let expectation = AttestationExpectation {
        purpose: &envelope.purpose,
        subject_sha256: &envelope.subject_sha256,
        payload_sha256: &envelope.payload_sha256,
        context_sha256: envelope.context_sha256.as_ref(),
    };
    let verified = verify_attestation_authority(
        envelope.clone(), expectation, &AttestationPolicy::default(), &Verifier,
        AttestationTrustContext { evaluation_time_unix_s: 105, snapshot, tracker: &tracker },
    ).unwrap();
    let proof = prove_root_role_quorum(root, directory, snapshot, &verified, role).unwrap();
    authorize_role_under_root(state, &proof).unwrap()
}

#[test]
fn independent_authorized_sources_establish_bounded_trusted_time() {
    let (root, directory, snapshot, state) = fixture();
    let log_statement = TimeSourceStatement::new(
        "log-a", TimeEvidenceKind::TransparencyIntegrated, 100, 110,
        Sha256Digest::of_bytes(b"log-checkpoint"),
    ).unwrap();
    let witness_statement = TimeSourceStatement::new(
        "witness-b", TimeEvidenceKind::WitnessedCheckpoint, 105, 115,
        Sha256Digest::of_bytes(b"witness-checkpoint"),
    ).unwrap();
    let log_authority = authorize_statement(
        &root, &directory, &snapshot, &state, TrustRole::TransparencyLog, "log-key", &log_statement,
    );
    let witness_authority = authorize_statement(
        &root, &directory, &snapshot, &state, TrustRole::TransparencyWitness, "witness-key", &witness_statement,
    );
    let evidence = vec![
        log_statement.bind_authority(log_authority.authority_sha256().clone()).unwrap(),
        witness_statement.bind_authority(witness_authority.authority_sha256().clone()).unwrap(),
    ];
    let policy = TimeAssessmentPolicy {
        minimum_distinct_sources: 2,
        minimum_kind: TimeEvidenceKind::TransparencyIntegrated,
        maximum_consensus_width_s: 10,
    };
    let assessment = assess_time(policy.clone(), &evidence);
    let trusted = establish_trusted_time(
        policy, &assessment, &evidence, &[log_authority, witness_authority],
    ).unwrap();
    assert_eq!(trusted.consensus_interval(), (105, 110));
    assert!(trusted.trusted_time_established());
    assert!(!trusted.current_time_established());
}

#[test]
fn authority_cannot_be_reused_for_a_different_time_statement() {
    let (root, directory, snapshot, state) = fixture();
    let first = TimeSourceStatement::new(
        "log-a", TimeEvidenceKind::TransparencyIntegrated, 100, 110,
        Sha256Digest::of_bytes(b"log-checkpoint"),
    ).unwrap();
    let authority = authorize_statement(
        &root, &directory, &snapshot, &state, TrustRole::TransparencyLog, "log-key", &first,
    );
    let altered = TimeSourceStatement::new(
        "log-a", TimeEvidenceKind::TransparencyIntegrated, 100, 120,
        Sha256Digest::of_bytes(b"log-checkpoint"),
    ).unwrap();
    let evidence = vec![altered.bind_authority(authority.authority_sha256().clone()).unwrap()];
    let policy = TimeAssessmentPolicy {
        minimum_distinct_sources: 1,
        minimum_kind: TimeEvidenceKind::TransparencyIntegrated,
        maximum_consensus_width_s: 30,
    };
    let assessment = assess_time(policy.clone(), &evidence);
    let issues = establish_trusted_time(policy, &assessment, &evidence, &[authority]).unwrap_err();
    assert!(issues.iter().any(|issue| matches!(issue, TrustedTimeIssue::SourceStatementMismatch { .. })));
}
