// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Role-aware authority over frozen trust roots and trust snapshots.
//!
//! `VerifiedAttestation` proves cryptography and lifecycle against one exact
//! tracker-current `TrustSnapshot`. This module adds the missing institutional
//! layer: counted signers must resolve through the exact root-bound principal
//! directory and satisfy the root's principal/organization/region/algorithm
//! geometry. Ordinary role authority additionally requires that the snapshot
//! used by the attestation is itself authorized under the same root.
//!
//! Genesis remains an explicit external bootstrap boundary. Root rotation
//! requires an authorized old root/snapshot, an old-root quorum over the exact
//! successor, a self-consistent new-root quorum, and an exact candidate successor
//! snapshot that becomes authorized only as part of the successful transition.

use std::collections::BTreeSet;

use serde::Serialize;

use crate::{
    FramedDigest, FrozenTrustRoot, PrincipalResolutionError, Sha256Digest,
    SignatureAlgorithm, TrustRole, TrustRootTransitionContract, TrustSnapshot,
    TrustUsage, TrustedPrincipalDirectory, VerifiedAttestation,
};

const ROOT_ROLE_PROOF_DOMAIN: &str = "symthaea.root-role-quorum-proof.identity.v1";
const GENESIS_ROOT_AUTHORITY_DOMAIN: &str = "symthaea.genesis-root-authority.identity.v2";
const TRUST_SNAPSHOT_AUTHORITY_DOMAIN: &str = "symthaea.authorized-trust-snapshot.identity.v1";
const ROOT_TRANSITION_AUTHORITY_DOMAIN: &str = "symthaea.root-transition-authority.identity.v2";
const AUTHORIZED_ROLE_DOMAIN: &str = "symthaea.authorized-root-role.identity.v2";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RoleSignerIdentity {
    pub algorithm: SignatureAlgorithm,
    pub key_id: String,
    pub verification_key_sha256: Sha256Digest,
    pub principal_id: String,
    pub organization_id: String,
    pub region_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RootRoleQuorumFinding {
    PrincipalDirectoryMismatch,
    TrustSnapshotIdentityUnavailable,
    TrustSnapshotMismatch,
    RootOutsideValidityWindow,
    MissingRolePolicy,
    PurposeMismatch,
    SignerMissingFromTrustSnapshot {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
    PrincipalResolutionFailed {
        algorithm: SignatureAlgorithm,
        key_id: String,
        reason: PrincipalResolutionError,
    },
    PrincipalNotAllowed {
        principal_id: String,
    },
    InsufficientValidSignatures {
        actual: usize,
        required: usize,
    },
    InsufficientDistinctPrincipals {
        actual: usize,
        required: usize,
    },
    InsufficientDistinctOrganizations {
        actual: usize,
        required: usize,
    },
    InsufficientDistinctRegions {
        actual: usize,
        required: usize,
    },
    MissingRequiredAlgorithm {
        algorithm: SignatureAlgorithm,
    },
}

/// Structural proof that one already-verified attestation also satisfies one
/// frozen root role. It intentionally does not establish authority by itself:
/// the bound trust snapshot may still be merely a caller-presented candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RootRoleQuorumProof {
    role: TrustRole,
    root_sha256: Sha256Digest,
    principal_directory_sha256: Sha256Digest,
    trust_snapshot_sha256: Sha256Digest,
    attestation_authority_sha256: Sha256Digest,
    attestation_sha256: Sha256Digest,
    subject_sha256: Sha256Digest,
    payload_sha256: Sha256Digest,
    context_sha256: Option<Sha256Digest>,
    evaluation_time_unix_s: u64,
    signers: Vec<RoleSignerIdentity>,
    proof_sha256: Sha256Digest,
}

impl RootRoleQuorumProof {
    pub fn role(&self) -> TrustRole {
        self.role
    }

    pub fn root_sha256(&self) -> &Sha256Digest {
        &self.root_sha256
    }

    pub fn principal_directory_sha256(&self) -> &Sha256Digest {
        &self.principal_directory_sha256
    }

    pub fn trust_snapshot_sha256(&self) -> &Sha256Digest {
        &self.trust_snapshot_sha256
    }

    pub fn attestation_authority_sha256(&self) -> &Sha256Digest {
        &self.attestation_authority_sha256
    }

    pub fn subject_sha256(&self) -> &Sha256Digest {
        &self.subject_sha256
    }

    pub fn payload_sha256(&self) -> &Sha256Digest {
        &self.payload_sha256
    }

    pub fn context_sha256(&self) -> Option<&Sha256Digest> {
        self.context_sha256.as_ref()
    }

    pub fn evaluation_time_unix_s(&self) -> u64 {
        self.evaluation_time_unix_s
    }

    pub fn signers(&self) -> &[RoleSignerIdentity] {
        &self.signers
    }

    pub fn proof_sha256(&self) -> &Sha256Digest {
        &self.proof_sha256
    }
}

pub fn prove_root_role_quorum(
    root: &FrozenTrustRoot,
    directory: &TrustedPrincipalDirectory,
    snapshot: &TrustSnapshot,
    verified: &VerifiedAttestation,
    role: TrustRole,
) -> Result<RootRoleQuorumProof, Vec<RootRoleQuorumFinding>> {
    let mut findings = Vec::new();

    if root.principal_directory_sequence() != directory.sequence()
        || root.principal_directory_sha256() != directory.directory_sha256()
    {
        findings.push(RootRoleQuorumFinding::PrincipalDirectoryMismatch);
    }

    let snapshot_sha256 = match snapshot.digest() {
        Ok(digest) => {
            if &digest != verified.trust_snapshot_sha256() {
                findings.push(RootRoleQuorumFinding::TrustSnapshotMismatch);
            }
            digest
        }
        Err(_) => {
            findings.push(RootRoleQuorumFinding::TrustSnapshotIdentityUnavailable);
            Sha256Digest::of_bytes(b"invalid-trust-snapshot")
        }
    };

    let evaluation_time = verified.evaluation_time_unix_s();
    if evaluation_time < root.issued_at_unix_s() || evaluation_time >= root.expires_at_unix_s() {
        findings.push(RootRoleQuorumFinding::RootOutsideValidityWindow);
    }

    let Some(policy) = root.role_policy(role) else {
        findings.push(RootRoleQuorumFinding::MissingRolePolicy);
        return Err(findings);
    };

    let expected_usage = role_usage(role);
    if verified.envelope().purpose != expected_usage {
        findings.push(RootRoleQuorumFinding::PurposeMismatch);
    }

    let mut signers = Vec::new();
    let mut principals = BTreeSet::new();
    let mut organizations = BTreeSet::new();
    let mut regions = BTreeSet::new();
    let mut algorithms = BTreeSet::new();

    for (algorithm, key_id) in verified.valid_signers() {
        let Some(key_record) = snapshot.key_record(algorithm, key_id) else {
            findings.push(RootRoleQuorumFinding::SignerMissingFromTrustSnapshot {
                algorithm: algorithm.clone(),
                key_id: key_id.clone(),
            });
            continue;
        };

        let resolved = match directory.resolve_key_for_role(
            algorithm,
            key_id,
            &key_record.verification_key_sha256,
            role,
        ) {
            Ok(resolved) => resolved,
            Err(reason) => {
                findings.push(RootRoleQuorumFinding::PrincipalResolutionFailed {
                    algorithm: algorithm.clone(),
                    key_id: key_id.clone(),
                    reason,
                });
                continue;
            }
        };

        if policy
            .allowed_principal_ids
            .as_ref()
            .is_some_and(|allowed| !allowed.contains(resolved.principal_id()))
        {
            findings.push(RootRoleQuorumFinding::PrincipalNotAllowed {
                principal_id: resolved.principal_id().to_string(),
            });
            continue;
        }

        principals.insert(resolved.principal_id().to_string());
        organizations.insert(resolved.organization_id().to_string());
        regions.insert(resolved.region_id().to_string());
        algorithms.insert(algorithm.clone());
        signers.push(RoleSignerIdentity {
            algorithm: algorithm.clone(),
            key_id: key_id.clone(),
            verification_key_sha256: key_record.verification_key_sha256.clone(),
            principal_id: resolved.principal_id().to_string(),
            organization_id: resolved.organization_id().to_string(),
            region_id: resolved.region_id().to_string(),
        });
    }

    if signers.len() < policy.minimum_valid_signatures {
        findings.push(RootRoleQuorumFinding::InsufficientValidSignatures {
            actual: signers.len(),
            required: policy.minimum_valid_signatures,
        });
    }
    if principals.len() < policy.minimum_distinct_principals {
        findings.push(RootRoleQuorumFinding::InsufficientDistinctPrincipals {
            actual: principals.len(),
            required: policy.minimum_distinct_principals,
        });
    }
    if organizations.len() < policy.minimum_distinct_organizations {
        findings.push(RootRoleQuorumFinding::InsufficientDistinctOrganizations {
            actual: organizations.len(),
            required: policy.minimum_distinct_organizations,
        });
    }
    if regions.len() < policy.minimum_distinct_regions {
        findings.push(RootRoleQuorumFinding::InsufficientDistinctRegions {
            actual: regions.len(),
            required: policy.minimum_distinct_regions,
        });
    }
    for algorithm in &policy.required_algorithms {
        if !algorithms.contains(algorithm) {
            findings.push(RootRoleQuorumFinding::MissingRequiredAlgorithm {
                algorithm: algorithm.clone(),
            });
        }
    }

    if !findings.is_empty() {
        return Err(findings);
    }

    signers.sort_by(|left, right| {
        left.principal_id
            .cmp(&right.principal_id)
            .then(left.algorithm.cmp(&right.algorithm))
            .then(left.key_id.cmp(&right.key_id))
    });

    let proof_sha256 = role_quorum_digest(
        role,
        root.root_sha256(),
        directory.directory_sha256(),
        &snapshot_sha256,
        verified,
        &signers,
    );
    Ok(RootRoleQuorumProof {
        role,
        root_sha256: root.root_sha256().clone(),
        principal_directory_sha256: directory.directory_sha256().clone(),
        trust_snapshot_sha256: snapshot_sha256,
        attestation_authority_sha256: verified.authority_sha256().clone(),
        attestation_sha256: verified.attestation_sha256().clone(),
        subject_sha256: verified.envelope().subject_sha256.clone(),
        payload_sha256: verified.envelope().payload_sha256.clone(),
        context_sha256: verified.envelope().context_sha256.clone(),
        evaluation_time_unix_s: evaluation_time,
        signers,
        proof_sha256,
    })
}

pub trait GenesisTrustAnchorVerifier {
    /// Stable identity of the external bootstrap authority/configuration used by
    /// this verifier. The implementation is part of the embedding trust base.
    fn verifier_authority_sha256(&self) -> Sha256Digest;

    fn verify_genesis_state(
        &self,
        root_sha256: &Sha256Digest,
        principal_directory_sha256: &Sha256Digest,
        initial_trust_snapshot_sha256: &Sha256Digest,
        anchor_artifact_sha256: &Sha256Digest,
        anchored_at_unix_s: u64,
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenesisTrustAnchorEvidence {
    pub anchor_artifact_sha256: Sha256Digest,
    pub initial_trust_snapshot_sha256: Sha256Digest,
    pub anchored_at_unix_s: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenesisRootAuthorizationError {
    NotGenesisRoot,
    PrincipalDirectoryMismatch,
    InitialTrustSnapshotIdentityUnavailable,
    InitialTrustSnapshotMismatch,
    InitialTrustSnapshotNotFresh,
    AnchorOutsideRootWindow,
    VerificationProviderError(String),
    AnchorRejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RootAuthorizationKind {
    Genesis,
    Transition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum TrustSnapshotAuthorizationKind {
    GenesisBootstrap,
    RootTransition,
}

/// Non-forgeable authority over one exact frozen trust root. Serializable for
/// evidence retention, intentionally not deserializable into authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AuthorizedTrustRoot {
    root_sha256: Sha256Digest,
    principal_directory_sha256: Sha256Digest,
    root_version: u64,
    authorized_at_unix_s: u64,
    authorization_kind: RootAuthorizationKind,
    authority_sha256: Sha256Digest,
}

impl AuthorizedTrustRoot {
    pub fn root_sha256(&self) -> &Sha256Digest {
        &self.root_sha256
    }

    pub fn principal_directory_sha256(&self) -> &Sha256Digest {
        &self.principal_directory_sha256
    }

    pub fn root_version(&self) -> u64 {
        self.root_version
    }

    pub fn authorized_at_unix_s(&self) -> u64 {
        self.authorized_at_unix_s
    }

    pub fn authorization_kind(&self) -> RootAuthorizationKind {
        self.authorization_kind
    }

    pub fn authority_sha256(&self) -> &Sha256Digest {
        &self.authority_sha256
    }

    pub const fn current_root_established(&self) -> bool {
        false
    }
}

/// Authority over one exact trust snapshot under one exact authorized root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AuthorizedTrustSnapshot {
    root_authority_sha256: Sha256Digest,
    root_sha256: Sha256Digest,
    trust_snapshot_sha256: Sha256Digest,
    trust_snapshot_sequence: u64,
    authorized_at_unix_s: u64,
    authorization_kind: TrustSnapshotAuthorizationKind,
    authority_sha256: Sha256Digest,
}

impl AuthorizedTrustSnapshot {
    pub fn root_authority_sha256(&self) -> &Sha256Digest {
        &self.root_authority_sha256
    }

    pub fn root_sha256(&self) -> &Sha256Digest {
        &self.root_sha256
    }

    pub fn trust_snapshot_sha256(&self) -> &Sha256Digest {
        &self.trust_snapshot_sha256
    }

    pub fn trust_snapshot_sequence(&self) -> u64 {
        self.trust_snapshot_sequence
    }

    pub fn authorized_at_unix_s(&self) -> u64 {
        self.authorized_at_unix_s
    }

    pub fn authorization_kind(&self) -> TrustSnapshotAuthorizationKind {
        self.authorization_kind
    }

    pub fn authority_sha256(&self) -> &Sha256Digest {
        &self.authority_sha256
    }

    pub const fn current_snapshot_established(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AuthorizedTrustState {
    root: AuthorizedTrustRoot,
    snapshot: AuthorizedTrustSnapshot,
}

impl AuthorizedTrustState {
    pub fn root(&self) -> &AuthorizedTrustRoot {
        &self.root
    }

    pub fn snapshot(&self) -> &AuthorizedTrustSnapshot {
        &self.snapshot
    }

    pub const fn current_state_established(&self) -> bool {
        false
    }
}

pub fn authorize_genesis_trust_state(
    root: &FrozenTrustRoot,
    directory: &TrustedPrincipalDirectory,
    initial_snapshot: &TrustSnapshot,
    anchor: &GenesisTrustAnchorEvidence,
    verifier: &dyn GenesisTrustAnchorVerifier,
) -> Result<AuthorizedTrustState, GenesisRootAuthorizationError> {
    if root.version() != 1 || root.predecessor_root_sha256().is_some() {
        return Err(GenesisRootAuthorizationError::NotGenesisRoot);
    }
    if root.principal_directory_sequence() != directory.sequence()
        || root.principal_directory_sha256() != directory.directory_sha256()
    {
        return Err(GenesisRootAuthorizationError::PrincipalDirectoryMismatch);
    }
    if anchor.anchored_at_unix_s < root.issued_at_unix_s()
        || anchor.anchored_at_unix_s >= root.expires_at_unix_s()
    {
        return Err(GenesisRootAuthorizationError::AnchorOutsideRootWindow);
    }
    let snapshot_sha256 = initial_snapshot
        .digest()
        .map_err(|_| GenesisRootAuthorizationError::InitialTrustSnapshotIdentityUnavailable)?;
    if snapshot_sha256 != anchor.initial_trust_snapshot_sha256 {
        return Err(GenesisRootAuthorizationError::InitialTrustSnapshotMismatch);
    }
    if !initial_snapshot.is_fresh_at(anchor.anchored_at_unix_s) {
        return Err(GenesisRootAuthorizationError::InitialTrustSnapshotNotFresh);
    }
    let accepted = verifier
        .verify_genesis_state(
            root.root_sha256(),
            directory.directory_sha256(),
            &snapshot_sha256,
            &anchor.anchor_artifact_sha256,
            anchor.anchored_at_unix_s,
        )
        .map_err(GenesisRootAuthorizationError::VerificationProviderError)?;
    if !accepted {
        return Err(GenesisRootAuthorizationError::AnchorRejected);
    }

    let verifier_authority_sha256 = verifier.verifier_authority_sha256();
    let root_authority_sha256 = genesis_authority_digest(
        root.root_sha256(),
        directory.directory_sha256(),
        &snapshot_sha256,
        &anchor.anchor_artifact_sha256,
        &verifier_authority_sha256,
        anchor.anchored_at_unix_s,
    );
    let root_authority = AuthorizedTrustRoot {
        root_sha256: root.root_sha256().clone(),
        principal_directory_sha256: directory.directory_sha256().clone(),
        root_version: root.version(),
        authorized_at_unix_s: anchor.anchored_at_unix_s,
        authorization_kind: RootAuthorizationKind::Genesis,
        authority_sha256: root_authority_sha256,
    };
    let snapshot_authority_sha256 = trust_snapshot_authority_digest(
        root_authority.authority_sha256(),
        &snapshot_sha256,
        initial_snapshot.sequence,
        anchor.anchored_at_unix_s,
        TrustSnapshotAuthorizationKind::GenesisBootstrap,
    );
    let snapshot_authority = AuthorizedTrustSnapshot {
        root_authority_sha256: root_authority.authority_sha256().clone(),
        root_sha256: root.root_sha256().clone(),
        trust_snapshot_sha256: snapshot_sha256,
        trust_snapshot_sequence: initial_snapshot.sequence,
        authorized_at_unix_s: anchor.anchored_at_unix_s,
        authorization_kind: TrustSnapshotAuthorizationKind::GenesisBootstrap,
        authority_sha256: snapshot_authority_sha256,
    };
    Ok(AuthorizedTrustState {
        root: root_authority,
        snapshot: snapshot_authority,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthorizedRoleError {
    RootMismatch,
    PrincipalDirectoryMismatch,
    TrustSnapshotMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AuthorizedTrustRoleAttestation {
    root_authority_sha256: Sha256Digest,
    trust_snapshot_authority_sha256: Sha256Digest,
    role: TrustRole,
    role_quorum_proof_sha256: Sha256Digest,
    attestation_authority_sha256: Sha256Digest,
    subject_sha256: Sha256Digest,
    payload_sha256: Sha256Digest,
    context_sha256: Option<Sha256Digest>,
    authority_sha256: Sha256Digest,
}

impl AuthorizedTrustRoleAttestation {
    pub fn role(&self) -> TrustRole {
        self.role
    }

    pub fn root_authority_sha256(&self) -> &Sha256Digest {
        &self.root_authority_sha256
    }

    pub fn trust_snapshot_authority_sha256(&self) -> &Sha256Digest {
        &self.trust_snapshot_authority_sha256
    }

    pub fn role_quorum_proof_sha256(&self) -> &Sha256Digest {
        &self.role_quorum_proof_sha256
    }

    pub fn attestation_authority_sha256(&self) -> &Sha256Digest {
        &self.attestation_authority_sha256
    }

    pub fn subject_sha256(&self) -> &Sha256Digest {
        &self.subject_sha256
    }

    pub fn payload_sha256(&self) -> &Sha256Digest {
        &self.payload_sha256
    }

    pub fn context_sha256(&self) -> Option<&Sha256Digest> {
        self.context_sha256.as_ref()
    }

    pub fn authority_sha256(&self) -> &Sha256Digest {
        &self.authority_sha256
    }
}

pub fn authorize_role_under_root(
    state: &AuthorizedTrustState,
    proof: &RootRoleQuorumProof,
) -> Result<AuthorizedTrustRoleAttestation, AuthorizedRoleError> {
    if proof.root_sha256() != state.root.root_sha256() {
        return Err(AuthorizedRoleError::RootMismatch);
    }
    if proof.principal_directory_sha256() != state.root.principal_directory_sha256() {
        return Err(AuthorizedRoleError::PrincipalDirectoryMismatch);
    }
    if proof.trust_snapshot_sha256() != state.snapshot.trust_snapshot_sha256() {
        return Err(AuthorizedRoleError::TrustSnapshotMismatch);
    }
    let authority_sha256 = authorized_role_digest(state, proof);
    Ok(AuthorizedTrustRoleAttestation {
        root_authority_sha256: state.root.authority_sha256().clone(),
        trust_snapshot_authority_sha256: state.snapshot.authority_sha256().clone(),
        role: proof.role(),
        role_quorum_proof_sha256: proof.proof_sha256().clone(),
        attestation_authority_sha256: proof.attestation_authority_sha256().clone(),
        subject_sha256: proof.subject_sha256().clone(),
        payload_sha256: proof.payload_sha256().clone(),
        context_sha256: proof.context_sha256().cloned(),
        authority_sha256,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RootTransitionAuthorizationError {
    PreviousAuthorityMismatch,
    PreviousSnapshotRootMismatch,
    TransitionContractMismatch,
    NextTrustSnapshotIdentityUnavailable,
    NextTrustSnapshotNotFresh,
    TrustSnapshotSequenceNotAdvanced { previous: u64, next: u64 },
    OldProofWrongRole,
    NewProofWrongRole,
    OldProofWrongRoot,
    NewProofWrongRoot,
    OldProofWrongTrustSnapshot,
    NewProofWrongTrustSnapshot,
    OldProofWrongTarget,
    NewProofWrongTarget,
    OldProofTimeMismatch,
    NewProofTimeMismatch,
}

pub fn authorize_root_transition(
    previous_state: &AuthorizedTrustState,
    previous_root: &FrozenTrustRoot,
    next_root: &FrozenTrustRoot,
    next_snapshot: &TrustSnapshot,
    transition_at_unix_s: u64,
    contract: &TrustRootTransitionContract,
    old_root_proof: &RootRoleQuorumProof,
    new_root_proof: &RootRoleQuorumProof,
) -> Result<AuthorizedTrustState, RootTransitionAuthorizationError> {
    if previous_state.root.root_sha256() != previous_root.root_sha256() {
        return Err(RootTransitionAuthorizationError::PreviousAuthorityMismatch);
    }
    if previous_state.snapshot.root_sha256() != previous_root.root_sha256()
        || previous_state.snapshot.root_authority_sha256()
            != previous_state.root.authority_sha256()
    {
        return Err(RootTransitionAuthorizationError::PreviousSnapshotRootMismatch);
    }

    let next_snapshot_sha256 = next_snapshot
        .digest()
        .map_err(|_| RootTransitionAuthorizationError::NextTrustSnapshotIdentityUnavailable)?;
    if !next_snapshot.is_fresh_at(transition_at_unix_s) {
        return Err(RootTransitionAuthorizationError::NextTrustSnapshotNotFresh);
    }
    if next_snapshot.sequence <= previous_state.snapshot.trust_snapshot_sequence() {
        return Err(RootTransitionAuthorizationError::TrustSnapshotSequenceNotAdvanced {
            previous: previous_state.snapshot.trust_snapshot_sequence(),
            next: next_snapshot.sequence,
        });
    }

    let reconstructed = TrustRootTransitionContract::new(
        previous_root,
        next_root,
        transition_at_unix_s,
        old_root_proof.proof_sha256().clone(),
        new_root_proof.proof_sha256().clone(),
    )
    .map_err(|_| RootTransitionAuthorizationError::TransitionContractMismatch)?;
    if reconstructed.transition_sha256() != contract.transition_sha256() {
        return Err(RootTransitionAuthorizationError::TransitionContractMismatch);
    }
    if old_root_proof.role() != TrustRole::Root {
        return Err(RootTransitionAuthorizationError::OldProofWrongRole);
    }
    if new_root_proof.role() != TrustRole::Root {
        return Err(RootTransitionAuthorizationError::NewProofWrongRole);
    }
    if old_root_proof.root_sha256() != previous_root.root_sha256() {
        return Err(RootTransitionAuthorizationError::OldProofWrongRoot);
    }
    if new_root_proof.root_sha256() != next_root.root_sha256() {
        return Err(RootTransitionAuthorizationError::NewProofWrongRoot);
    }
    if old_root_proof.trust_snapshot_sha256()
        != previous_state.snapshot.trust_snapshot_sha256()
    {
        return Err(RootTransitionAuthorizationError::OldProofWrongTrustSnapshot);
    }
    if new_root_proof.trust_snapshot_sha256() != &next_snapshot_sha256 {
        return Err(RootTransitionAuthorizationError::NewProofWrongTrustSnapshot);
    }

    let expected_subject = next_root.root_sha256();
    let expected_payload = next_root.principal_directory_sha256();
    let expected_context = Some(previous_root.root_sha256());
    if old_root_proof.subject_sha256() != expected_subject
        || old_root_proof.payload_sha256() != expected_payload
        || old_root_proof.context_sha256() != expected_context
    {
        return Err(RootTransitionAuthorizationError::OldProofWrongTarget);
    }
    if new_root_proof.subject_sha256() != expected_subject
        || new_root_proof.payload_sha256() != expected_payload
        || new_root_proof.context_sha256() != expected_context
    {
        return Err(RootTransitionAuthorizationError::NewProofWrongTarget);
    }
    if old_root_proof.evaluation_time_unix_s() != transition_at_unix_s {
        return Err(RootTransitionAuthorizationError::OldProofTimeMismatch);
    }
    if new_root_proof.evaluation_time_unix_s() != transition_at_unix_s {
        return Err(RootTransitionAuthorizationError::NewProofTimeMismatch);
    }

    let root_authority_sha256 = root_transition_authority_digest(
        previous_state,
        contract,
        old_root_proof,
        new_root_proof,
        next_root,
        &next_snapshot_sha256,
        transition_at_unix_s,
    );
    let root_authority = AuthorizedTrustRoot {
        root_sha256: next_root.root_sha256().clone(),
        principal_directory_sha256: next_root.principal_directory_sha256().clone(),
        root_version: next_root.version(),
        authorized_at_unix_s: transition_at_unix_s,
        authorization_kind: RootAuthorizationKind::Transition,
        authority_sha256: root_authority_sha256,
    };
    let snapshot_authority_sha256 = trust_snapshot_authority_digest(
        root_authority.authority_sha256(),
        &next_snapshot_sha256,
        next_snapshot.sequence,
        transition_at_unix_s,
        TrustSnapshotAuthorizationKind::RootTransition,
    );
    let snapshot_authority = AuthorizedTrustSnapshot {
        root_authority_sha256: root_authority.authority_sha256().clone(),
        root_sha256: next_root.root_sha256().clone(),
        trust_snapshot_sha256: next_snapshot_sha256,
        trust_snapshot_sequence: next_snapshot.sequence,
        authorized_at_unix_s: transition_at_unix_s,
        authorization_kind: TrustSnapshotAuthorizationKind::RootTransition,
        authority_sha256: snapshot_authority_sha256,
    };
    Ok(AuthorizedTrustState {
        root: root_authority,
        snapshot: snapshot_authority,
    })
}

fn role_usage(role: TrustRole) -> TrustUsage {
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
    TrustUsage::parse(format!("trust.role.{suffix}"))
        .expect("fixed role usage identifiers are canonical")
}

fn role_quorum_digest(
    role: TrustRole,
    root_sha256: &Sha256Digest,
    directory_sha256: &Sha256Digest,
    trust_snapshot_sha256: &Sha256Digest,
    verified: &VerifiedAttestation,
    signers: &[RoleSignerIdentity],
) -> Sha256Digest {
    let mut digest = FramedDigest::new(ROOT_ROLE_PROOF_DOMAIN);
    digest.text(role_tag(role));
    digest.text(root_sha256.as_str());
    digest.text(directory_sha256.as_str());
    digest.text(trust_snapshot_sha256.as_str());
    digest.text(verified.authority_sha256().as_str());
    digest.text(verified.attestation_sha256().as_str());
    digest.text(verified.envelope().subject_sha256.as_str());
    digest.text(verified.envelope().payload_sha256.as_str());
    digest.optional_sha(verified.envelope().context_sha256.as_ref());
    digest.text(&verified.evaluation_time_unix_s().to_string());
    for signer in signers {
        digest.text("signer");
        digest_algorithm(&mut digest, &signer.algorithm);
        digest.text(&signer.key_id);
        digest.text(signer.verification_key_sha256.as_str());
        digest.text(&signer.principal_id);
        digest.text(&signer.organization_id);
        digest.text(&signer.region_id);
    }
    digest.digest()
}

fn genesis_authority_digest(
    root_sha256: &Sha256Digest,
    directory_sha256: &Sha256Digest,
    initial_snapshot_sha256: &Sha256Digest,
    anchor_artifact_sha256: &Sha256Digest,
    verifier_authority_sha256: &Sha256Digest,
    anchored_at_unix_s: u64,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(GENESIS_ROOT_AUTHORITY_DOMAIN);
    digest.text(root_sha256.as_str());
    digest.text(directory_sha256.as_str());
    digest.text(initial_snapshot_sha256.as_str());
    digest.text(anchor_artifact_sha256.as_str());
    digest.text(verifier_authority_sha256.as_str());
    digest.text(&anchored_at_unix_s.to_string());
    digest.digest()
}

fn trust_snapshot_authority_digest(
    root_authority_sha256: &Sha256Digest,
    snapshot_sha256: &Sha256Digest,
    sequence: u64,
    authorized_at_unix_s: u64,
    kind: TrustSnapshotAuthorizationKind,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(TRUST_SNAPSHOT_AUTHORITY_DOMAIN);
    digest.text(root_authority_sha256.as_str());
    digest.text(snapshot_sha256.as_str());
    digest.text(&sequence.to_string());
    digest.text(&authorized_at_unix_s.to_string());
    digest.text(match kind {
        TrustSnapshotAuthorizationKind::GenesisBootstrap => "genesis-bootstrap",
        TrustSnapshotAuthorizationKind::RootTransition => "root-transition",
    });
    digest.digest()
}

fn authorized_role_digest(
    state: &AuthorizedTrustState,
    proof: &RootRoleQuorumProof,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(AUTHORIZED_ROLE_DOMAIN);
    digest.text(state.root.authority_sha256().as_str());
    digest.text(state.snapshot.authority_sha256().as_str());
    digest.text(role_tag(proof.role()));
    digest.text(proof.proof_sha256().as_str());
    digest.text(proof.attestation_authority_sha256().as_str());
    digest.digest()
}

#[allow(clippy::too_many_arguments)]
fn root_transition_authority_digest(
    previous_state: &AuthorizedTrustState,
    contract: &TrustRootTransitionContract,
    old_root_proof: &RootRoleQuorumProof,
    new_root_proof: &RootRoleQuorumProof,
    next_root: &FrozenTrustRoot,
    next_snapshot_sha256: &Sha256Digest,
    transition_at_unix_s: u64,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(ROOT_TRANSITION_AUTHORITY_DOMAIN);
    digest.text(previous_state.root.authority_sha256().as_str());
    digest.text(previous_state.snapshot.authority_sha256().as_str());
    digest.text(contract.transition_sha256().as_str());
    digest.text(old_root_proof.proof_sha256().as_str());
    digest.text(new_root_proof.proof_sha256().as_str());
    digest.text(next_root.root_sha256().as_str());
    digest.text(next_root.principal_directory_sha256().as_str());
    digest.text(next_snapshot_sha256.as_str());
    digest.text(&transition_at_unix_s.to_string());
    digest.digest()
}

fn digest_algorithm(digest: &mut FramedDigest, algorithm: &SignatureAlgorithm) {
    match algorithm {
        SignatureAlgorithm::Ed25519 => digest.text("builtin:ed25519"),
        SignatureAlgorithm::MlDsa65 => digest.text("builtin:ml-dsa-65"),
        SignatureAlgorithm::MlDsa87 => digest.text("builtin:ml-dsa-87"),
        SignatureAlgorithm::Other(name) => {
            digest.text("other");
            digest.text(name);
        }
    }
}

const fn role_tag(role: TrustRole) -> &'static str {
    match role {
        TrustRole::Root => "root",
        TrustRole::Freshness => "freshness",
        TrustRole::KeyLifecycle => "key-lifecycle",
        TrustRole::QualificationProfile => "qualification-profile",
        TrustRole::QualificationDecision => "qualification-decision",
        TrustRole::QualificationLifecycle => "qualification-lifecycle",
        TrustRole::TransparencyLog => "transparency-log",
        TrustRole::TransparencyWitness => "transparency-witness",
        TrustRole::EmergencyRecovery => "emergency-recovery",
    }
}
