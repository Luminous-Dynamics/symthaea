// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;

use crate::*;

struct EchoSigner {
    key_id: &'static str,
}

impl AttestationSigner for EchoSigner {
    fn algorithm(&self) -> SignatureAlgorithm {
        SignatureAlgorithm::Ed25519
    }

    fn key_id(&self) -> &str {
        self.key_id
    }

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

struct BootstrapVerifier;

impl GenesisTrustAnchorVerifier for BootstrapVerifier {
    fn verifier_authority_sha256(&self) -> Sha256Digest {
        Sha256Digest::of_bytes(b"transition-test-bootstrap")
    }

    fn verify_genesis_state(
        &self,
        _root_sha256: &Sha256Digest,
        _principal_directory_sha256: &Sha256Digest,
        _initial_trust_snapshot_sha256: &Sha256Digest,
        _anchor_artifact_sha256: &Sha256Digest,
        _anchored_at_unix_s: u64,
    ) -> Result<bool, String> {
        Ok(true)
    }
}

fn key_material(key_id: &str) -> Sha256Digest {
    Sha256Digest::of_bytes(format!("verification-key:{key_id}").as_bytes())
}

fn root_usage() -> TrustUsage {
    TrustUsage::parse("trust.role.root").unwrap()
}

fn directory() -> TrustedPrincipalDirectory {
    TrustedPrincipalDirectory::new(
        1,
        100,
        vec![TrustedPrincipal {
            principal_id: "p-a".into(),
            organization_id: "org-a".into(),
            region_id: "region-a".into(),
            keys: vec![TrustedKeyBinding {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "a".into(),
                verification_key_sha256: key_material("a"),
                roles: BTreeSet::from([TrustRole::Root, TrustRole::Freshness]),
            }],
        }],
    )
    .unwrap()
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

fn genesis_root(directory: &TrustedPrincipalDirectory) -> FrozenTrustRoot {
    TrustRootDraft {
        version: 1,
        predecessor_root_sha256: None,
        issued_at_unix_s: 100,
        expires_at_unix_s: 900,
        role_policies: vec![role_policy(TrustRole::Root), role_policy(TrustRole::Freshness)],
    }
    .freeze(directory)
    .unwrap()
}

fn successor_root(
    previous: &FrozenTrustRoot,
    directory: &TrustedPrincipalDirectory,
) -> FrozenTrustRoot {
    TrustRootDraft {
        version: 2,
        predecessor_root_sha256: Some(previous.root_sha256().clone()),
        issued_at_unix_s: 400,
        expires_at_unix_s: 900,
        role_policies: vec![role_policy(TrustRole::Root), role_policy(TrustRole::Freshness)],
    }
    .freeze(directory)
    .unwrap()
}

fn snapshot(sequence: u64) -> TrustSnapshot {
    TrustSnapshot::new(
        sequence,
        100 + sequence,
        900,
        vec![KeyTrustRecord {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "a".into(),
            verification_key_sha256: key_material("a"),
            not_before_unix_s: 100,
            not_after_unix_s: Some(900),
            status: KeyLifecycleStatus::Active,
            usages: BTreeSet::from([root_usage()]),
        }],
    )
    .unwrap()
}

fn verified_transition_attestation(
    snapshot: &TrustSnapshot,
    next_root: &FrozenTrustRoot,
    previous_root: &FrozenTrustRoot,
) -> VerifiedAttestation {
    let signer = EchoSigner { key_id: "a" };
    let signers: [&dyn AttestationSigner; 1] = [&signer];
    let envelope = attest_digests(
        root_usage(),
        next_root.root_sha256().clone(),
        next_root.principal_directory_sha256().clone(),
        Some(previous_root.root_sha256().clone()),
        &signers,
    )
    .unwrap();
    let mut tracker = TrustSnapshotTracker::default();
    tracker.accept(snapshot).unwrap();
    let policy = AttestationPolicy {
        minimum_valid_signatures: 1,
        ..AttestationPolicy::default()
    };
    let expectation = AttestationExpectation {
        purpose: &envelope.purpose,
        subject_sha256: &envelope.subject_sha256,
        payload_sha256: &envelope.payload_sha256,
        context_sha256: envelope.context_sha256.as_ref(),
    };
    verify_attestation_authority(
        envelope.clone(),
        expectation,
        &policy,
        &EchoVerifier,
        AttestationTrustContext {
            evaluation_time_unix_s: 500,
            snapshot,
            tracker: &tracker,
        },
    )
    .unwrap()
}

struct TransitionFixture {
    previous_root: FrozenTrustRoot,
    next_root: FrozenTrustRoot,
    previous_state: AuthorizedTrustState,
    next_snapshot: TrustSnapshot,
    previous_feasibility: TrustRootSignatureFeasibilityProof,
    next_feasibility: TrustRootSignatureFeasibilityProof,
    old_root_proof: RootRoleQuorumProof,
    new_root_proof: RootRoleQuorumProof,
    contract: TrustRootTransitionContract,
}

fn transition_fixture() -> TransitionFixture {
    let directory = directory();
    let previous_root = genesis_root(&directory);
    let next_root = successor_root(&previous_root, &directory);
    let previous_snapshot = snapshot(1);
    let next_snapshot = snapshot(2);
    let previous_feasibility =
        prove_trust_root_signature_feasibility(&previous_root, &directory).unwrap();
    let next_feasibility =
        prove_trust_root_signature_feasibility(&next_root, &directory).unwrap();
    let previous_state = authorize_genesis_trust_state(
        &previous_root,
        &directory,
        &previous_feasibility,
        &previous_snapshot,
        &GenesisTrustAnchorEvidence {
            anchor_artifact_sha256: Sha256Digest::of_bytes(b"transition-test-anchor"),
            initial_trust_snapshot_sha256: previous_snapshot.digest().unwrap(),
            anchored_at_unix_s: 200,
        },
        &BootstrapVerifier,
    )
    .unwrap();

    let old_verified =
        verified_transition_attestation(&previous_snapshot, &next_root, &previous_root);
    let new_verified = verified_transition_attestation(&next_snapshot, &next_root, &previous_root);
    let old_root_proof = prove_root_role_quorum(
        &previous_root,
        &directory,
        &previous_snapshot,
        &old_verified,
        TrustRole::Root,
    )
    .unwrap();
    let new_root_proof = prove_root_role_quorum(
        &next_root,
        &directory,
        &next_snapshot,
        &new_verified,
        TrustRole::Root,
    )
    .unwrap();
    let contract = TrustRootTransitionContract::new(
        &previous_root,
        &next_root,
        500,
        old_root_proof.proof_sha256().clone(),
        new_root_proof.proof_sha256().clone(),
    )
    .unwrap();

    TransitionFixture {
        previous_root,
        next_root,
        previous_state,
        next_snapshot,
        previous_feasibility,
        next_feasibility,
        old_root_proof,
        new_root_proof,
        contract,
    }
}

#[test]
fn valid_transition_carries_next_root_feasibility_into_successor_state() {
    let fixture = transition_fixture();
    let next_state = authorize_root_transition(
        &fixture.previous_state,
        &fixture.previous_root,
        &fixture.previous_feasibility,
        &fixture.next_root,
        &fixture.next_feasibility,
        &fixture.next_snapshot,
        500,
        &fixture.contract,
        &fixture.old_root_proof,
        &fixture.new_root_proof,
    )
    .unwrap();

    assert_eq!(
        next_state.root_signature_feasibility_sha256(),
        fixture.next_feasibility.proof_sha256()
    );
    assert!(next_state.root_signature_feasibility_established());
    assert_ne!(
        next_state.authority_gate_sha256(),
        fixture.previous_state.authority_gate_sha256()
    );
    assert_eq!(
        next_state.root().root_sha256(),
        fixture.next_root.root_sha256()
    );
}

#[test]
fn previous_root_feasibility_cannot_be_substituted_with_next_root_proof() {
    let fixture = transition_fixture();
    let result = authorize_root_transition(
        &fixture.previous_state,
        &fixture.previous_root,
        &fixture.next_feasibility,
        &fixture.next_root,
        &fixture.next_feasibility,
        &fixture.next_snapshot,
        500,
        &fixture.contract,
        &fixture.old_root_proof,
        &fixture.new_root_proof,
    );

    assert_eq!(
        result,
        Err(RootTransitionTrustStateAuthorizationError::PreviousRootFeasibilityRootMismatch)
    );
}

#[test]
fn next_root_feasibility_cannot_be_substituted_with_previous_root_proof() {
    let fixture = transition_fixture();
    let result = authorize_root_transition(
        &fixture.previous_state,
        &fixture.previous_root,
        &fixture.previous_feasibility,
        &fixture.next_root,
        &fixture.previous_feasibility,
        &fixture.next_snapshot,
        500,
        &fixture.contract,
        &fixture.old_root_proof,
        &fixture.new_root_proof,
    );

    assert_eq!(
        result,
        Err(RootTransitionTrustStateAuthorizationError::NextRootFeasibilityRootMismatch)
    );
}
