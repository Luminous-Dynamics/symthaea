// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;

use crate::*;

struct EchoSigner { key_id: &'static str }
impl AttestationSigner for EchoSigner {
    fn algorithm(&self) -> SignatureAlgorithm { SignatureAlgorithm::Ed25519 }
    fn key_id(&self) -> &str { self.key_id }
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(Sha256Digest::of_bytes(message).as_str().as_bytes().to_vec())
    }
}

struct EchoVerifier;
impl AttestationSignatureVerifier for EchoVerifier {
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

struct BootstrapVerifier { identity: &'static str, accept: bool }
impl GenesisTrustAnchorVerifier for BootstrapVerifier {
    fn verifier_authority_sha256(&self) -> Sha256Digest {
        Sha256Digest::of_bytes(self.identity.as_bytes())
    }
    fn verify_genesis_state(
        &self,
        _root_sha256: &Sha256Digest,
        _principal_directory_sha256: &Sha256Digest,
        _initial_trust_snapshot_sha256: &Sha256Digest,
        _anchor_artifact_sha256: &Sha256Digest,
        _anchored_at_unix_s: u64,
    ) -> Result<bool, String> { Ok(self.accept) }
}

fn root_usage() -> TrustUsage { TrustUsage::parse("trust.role.root").unwrap() }
fn key_material(key_id: &str) -> Sha256Digest {
    Sha256Digest::of_bytes(format!("verification-key:{key_id}").as_bytes())
}
fn key_binding(key_id: &str) -> TrustedKeyBinding {
    TrustedKeyBinding {
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: key_id.into(),
        verification_key_sha256: key_material(key_id),
        roles: BTreeSet::from([TrustRole::Root, TrustRole::Freshness]),
    }
}
fn principal(id: &str, org: &str, region: &str, key_ids: &[&str]) -> TrustedPrincipal {
    TrustedPrincipal {
        principal_id: id.into(), organization_id: org.into(), region_id: region.into(),
        keys: key_ids.iter().map(|key_id| key_binding(key_id)).collect(),
    }
}
fn directory(principals: Vec<TrustedPrincipal>) -> TrustedPrincipalDirectory {
    TrustedPrincipalDirectory::new(1, 100, principals).unwrap()
}
fn role_policy(role: TrustRole, sigs: usize, principals: usize, orgs: usize, regions: usize) -> TrustRolePolicy {
    TrustRolePolicy {
        role,
        minimum_valid_signatures: sigs,
        minimum_distinct_principals: principals,
        minimum_distinct_organizations: orgs,
        minimum_distinct_regions: regions,
        required_algorithms: BTreeSet::from([SignatureAlgorithm::Ed25519]),
        allowed_principal_ids: None,
    }
}
fn genesis_root(directory: &TrustedPrincipalDirectory, root_policy: TrustRolePolicy) -> FrozenTrustRoot {
    TrustRootDraft {
        version: 1,
        predecessor_root_sha256: None,
        issued_at_unix_s: 100,
        expires_at_unix_s: 900,
        role_policies: vec![root_policy, role_policy(TrustRole::Freshness, 1, 1, 1, 1)],
    }.freeze(directory).unwrap()
}
fn trust_snapshot_with_sequence(key_ids: &[&str], sequence: u64) -> TrustSnapshot {
    TrustSnapshot::new(
        sequence, 100 + sequence, 900,
        key_ids.iter().map(|key_id| KeyTrustRecord {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: (*key_id).into(),
            verification_key_sha256: key_material(key_id),
            not_before_unix_s: 100,
            not_after_unix_s: Some(900),
            status: KeyLifecycleStatus::Active,
            usages: BTreeSet::from([root_usage()]),
        }).collect(),
    ).unwrap()
}
fn trust_snapshot(key_ids: &[&str]) -> TrustSnapshot { trust_snapshot_with_sequence(key_ids, 1) }

fn verified_root_attestation(
    snapshot: &TrustSnapshot,
    key_ids: &[&'static str],
    subject_sha256: Sha256Digest,
    payload_sha256: Sha256Digest,
    context_sha256: Option<Sha256Digest>,
) -> VerifiedAttestation {
    let signers: Vec<EchoSigner> = key_ids.iter().map(|key_id| EchoSigner { key_id: *key_id }).collect();
    let signer_refs: Vec<&dyn AttestationSigner> = signers.iter().map(|s| s as &dyn AttestationSigner).collect();
    let envelope = attest_digests(root_usage(), subject_sha256, payload_sha256, context_sha256, &signer_refs).unwrap();
    let mut tracker = TrustSnapshotTracker::default();
    tracker.accept(snapshot).unwrap();
    let policy = AttestationPolicy { minimum_valid_signatures: key_ids.len(), ..AttestationPolicy::default() };
    let expectation = AttestationExpectation {
        purpose: &envelope.purpose,
        subject_sha256: &envelope.subject_sha256,
        payload_sha256: &envelope.payload_sha256,
        context_sha256: envelope.context_sha256.as_ref(),
    };
    verify_attestation_authority(
        envelope.clone(), expectation, &policy, &EchoVerifier,
        AttestationTrustContext { evaluation_time_unix_s: 500, snapshot, tracker: &tracker },
    ).unwrap()
}

fn bootstrap_state(
    root: &FrozenTrustRoot,
    directory: &TrustedPrincipalDirectory,
    snapshot: &TrustSnapshot,
    provider: &'static str,
) -> AuthorizedTrustState {
    let anchor = GenesisTrustAnchorEvidence {
        anchor_artifact_sha256: Sha256Digest::of_bytes(b"offline-root-anchor"),
        initial_trust_snapshot_sha256: snapshot.digest().unwrap(),
        anchored_at_unix_s: 200,
    };
    authorize_genesis_trust_state(
        root, directory, snapshot, &anchor,
        &BootstrapVerifier { identity: provider, accept: true },
    ).unwrap()
}

#[test]
fn root_role_quorum_counts_principals_not_keys() {
    let directory = directory(vec![
        principal("p-a", "org-a", "region-a", &["a-1", "a-2"]),
        principal("p-b", "org-b", "region-b", &["b-1"]),
    ]);
    let root = genesis_root(&directory, role_policy(TrustRole::Root, 2, 2, 2, 2));
    let snapshot = trust_snapshot(&["a-1", "a-2", "b-1"]);
    let verified = verified_root_attestation(
        &snapshot, &["a-1", "a-2"], Sha256Digest::of_bytes(b"target"), Sha256Digest::of_bytes(b"payload"), None,
    );
    let findings = prove_root_role_quorum(&root, &directory, &snapshot, &verified, TrustRole::Root).unwrap_err();
    assert!(findings.iter().any(|f| matches!(f,
        RootRoleQuorumFinding::InsufficientDistinctPrincipals { actual: 1, required: 2 }
    )));
    assert!(findings.iter().any(|f| matches!(f,
        RootRoleQuorumFinding::InsufficientDistinctOrganizations { actual: 1, required: 2 }
    )));
}

#[test]
fn two_independent_principals_satisfy_root_quorum() {
    let directory = directory(vec![
        principal("p-a", "org-a", "region-a", &["a"]),
        principal("p-b", "org-b", "region-b", &["b"]),
    ]);
    let root = genesis_root(&directory, role_policy(TrustRole::Root, 2, 2, 2, 2));
    let snapshot = trust_snapshot(&["a", "b"]);
    let verified = verified_root_attestation(
        &snapshot, &["a", "b"], Sha256Digest::of_bytes(b"target"), Sha256Digest::of_bytes(b"payload"), None,
    );
    let proof = prove_root_role_quorum(&root, &directory, &snapshot, &verified, TrustRole::Root).unwrap();
    assert_eq!(proof.signers().len(), 2);
}

#[test]
fn bootstrap_provider_and_initial_snapshot_are_both_bound() {
    let directory = directory(vec![principal("p-a", "org-a", "region-a", &["a"])]);
    let root = genesis_root(&directory, role_policy(TrustRole::Root, 1, 1, 1, 1));
    let snapshot = trust_snapshot(&["a"]);
    let left = bootstrap_state(&root, &directory, &snapshot, "bootstrap-provider-a");
    let right = bootstrap_state(&root, &directory, &snapshot, "bootstrap-provider-b");
    assert_ne!(left.root().authority_sha256(), right.root().authority_sha256());
    assert_eq!(left.snapshot().trust_snapshot_sha256(), &snapshot.digest().unwrap());
}

#[test]
fn rejected_bootstrap_anchor_cannot_mint_trust_state() {
    let directory = directory(vec![principal("p-a", "org-a", "region-a", &["a"])]);
    let root = genesis_root(&directory, role_policy(TrustRole::Root, 1, 1, 1, 1));
    let snapshot = trust_snapshot(&["a"]);
    let result = authorize_genesis_trust_state(
        &root, &directory, &snapshot,
        &GenesisTrustAnchorEvidence {
            anchor_artifact_sha256: Sha256Digest::of_bytes(b"untrusted-anchor"),
            initial_trust_snapshot_sha256: snapshot.digest().unwrap(),
            anchored_at_unix_s: 200,
        },
        &BootstrapVerifier { identity: "bootstrap-provider", accept: false },
    );
    assert_eq!(result, Err(GenesisRootAuthorizationError::AnchorRejected));
}

#[test]
fn candidate_snapshot_cannot_substitute_for_authorized_snapshot() {
    let directory = directory(vec![principal("p-a", "org-a", "region-a", &["a"])]);
    let root = genesis_root(&directory, role_policy(TrustRole::Root, 1, 1, 1, 1));
    let authorized_snapshot = trust_snapshot_with_sequence(&["a"], 1);
    let state = bootstrap_state(&root, &directory, &authorized_snapshot, "bootstrap-provider");
    let candidate_snapshot = trust_snapshot_with_sequence(&["a"], 2);
    let verified = verified_root_attestation(
        &candidate_snapshot, &["a"], Sha256Digest::of_bytes(b"target"), Sha256Digest::of_bytes(b"payload"), None,
    );
    let proof = prove_root_role_quorum(
        &root, &directory, &candidate_snapshot, &verified, TrustRole::Root,
    ).unwrap();
    assert_eq!(
        authorize_role_under_root(&state, &proof),
        Err(AuthorizedRoleError::TrustSnapshotMismatch)
    );
}
