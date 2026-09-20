// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Feasibility-gated public trust-root authority surface.
//!
//! Historical raw root/snapshot authority capabilities remain implemented in
//! `root_authority`, but are no longer re-exported as public minting functions.
//! Public genesis and root-transition admission must carry the exact
//! `TrustRootSignatureFeasibilityProof` for the root being authorized. The
//! resulting public trust-state and role-attestation wrappers bind that gate
//! identity forward without changing historical `FrozenTrustRoot` hashes.

use serde::Serialize;

use crate::{
    FramedDigest, FrozenTrustRoot, GenesisRootAuthorizationError,
    GenesisTrustAnchorEvidence, GenesisTrustAnchorVerifier, RootRoleQuorumProof,
    RootTransitionAuthorizationError, Sha256Digest, TrustRootSignatureFeasibilityProof,
    TrustRootTransitionContract, TrustSnapshot, TrustedPrincipalDirectory,
};
use crate::root_authority::{
    AuthorizedRoleError,
    AuthorizedTrustRoleAttestation as RawAuthorizedTrustRoleAttestation,
    AuthorizedTrustRoot, AuthorizedTrustSnapshot,
    AuthorizedTrustState as RawAuthorizedTrustState,
    authorize_genesis_trust_state as authorize_genesis_trust_state_raw,
    authorize_role_under_root as authorize_role_under_root_raw,
    authorize_root_transition as authorize_root_transition_raw,
};

const TRUST_STATE_FEASIBILITY_GATE_DOMAIN: &str =
    "symthaea.feasibility-gated-trust-state.identity.v1";
const TRUST_ROLE_FEASIBILITY_GATE_DOMAIN: &str =
    "symthaea.feasibility-gated-root-role.identity.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenesisTrustStateAuthorizationError {
    RootFeasibilityRootMismatch,
    RootFeasibilityPrincipalDirectoryMismatch,
    Authorization(GenesisRootAuthorizationError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RootTransitionTrustStateAuthorizationError {
    PreviousRootFeasibilityRootMismatch,
    PreviousRootFeasibilityPrincipalDirectoryMismatch,
    PreviousRootFeasibilityProofMismatch,
    NextRootFeasibilityRootMismatch,
    NextRootFeasibilityPrincipalDirectoryMismatch,
    Authorization(RootTransitionAuthorizationError),
}

/// Public trust state whose construction proves that the exact root has enough
/// eligible role-bound keys to satisfy every configured signature threshold.
///
/// This is intentionally serialize-only: retained evidence cannot be
/// deserialized back into authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AuthorizedTrustState {
    inner: RawAuthorizedTrustState,
    root_signature_feasibility_sha256: Sha256Digest,
    authority_gate_sha256: Sha256Digest,
}

impl AuthorizedTrustState {
    pub fn root(&self) -> &AuthorizedTrustRoot {
        self.inner.root()
    }

    pub fn snapshot(&self) -> &AuthorizedTrustSnapshot {
        self.inner.snapshot()
    }

    pub fn root_signature_feasibility_sha256(&self) -> &Sha256Digest {
        &self.root_signature_feasibility_sha256
    }

    pub fn authority_gate_sha256(&self) -> &Sha256Digest {
        &self.authority_gate_sha256
    }

    pub const fn root_signature_feasibility_established(&self) -> bool {
        true
    }

    /// This gate establishes structural signature feasibility, not that this is
    /// the globally current trust state at an arbitrary later time.
    pub const fn current_state_established(&self) -> bool {
        false
    }
}

/// Public role authority whose identity includes the feasibility-gated trust
/// state that authorized the raw role proof.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AuthorizedTrustRoleAttestation {
    inner: RawAuthorizedTrustRoleAttestation,
    trust_state_gate_sha256: Sha256Digest,
    authority_sha256: Sha256Digest,
}

impl AuthorizedTrustRoleAttestation {
    pub fn role(&self) -> crate::TrustRole {
        self.inner.role()
    }

    pub fn root_authority_sha256(&self) -> &Sha256Digest {
        self.inner.root_authority_sha256()
    }

    pub fn trust_snapshot_authority_sha256(&self) -> &Sha256Digest {
        self.inner.trust_snapshot_authority_sha256()
    }

    pub fn role_quorum_proof_sha256(&self) -> &Sha256Digest {
        self.inner.role_quorum_proof_sha256()
    }

    pub fn attestation_authority_sha256(&self) -> &Sha256Digest {
        self.inner.attestation_authority_sha256()
    }

    pub fn subject_sha256(&self) -> &Sha256Digest {
        self.inner.subject_sha256()
    }

    pub fn payload_sha256(&self) -> &Sha256Digest {
        self.inner.payload_sha256()
    }

    pub fn context_sha256(&self) -> Option<&Sha256Digest> {
        self.inner.context_sha256()
    }

    pub fn trust_state_gate_sha256(&self) -> &Sha256Digest {
        &self.trust_state_gate_sha256
    }

    /// Public role-authority identity. Unlike the historical raw authority
    /// digest, this includes the feasibility-gated trust-state identity.
    pub fn authority_sha256(&self) -> &Sha256Digest {
        &self.authority_sha256
    }

    pub const fn root_signature_feasibility_established(&self) -> bool {
        true
    }
}

pub fn authorize_genesis_trust_state(
    root: &FrozenTrustRoot,
    directory: &TrustedPrincipalDirectory,
    root_feasibility: &TrustRootSignatureFeasibilityProof,
    initial_snapshot: &TrustSnapshot,
    anchor: &GenesisTrustAnchorEvidence,
    verifier: &dyn GenesisTrustAnchorVerifier,
) -> Result<AuthorizedTrustState, GenesisTrustStateAuthorizationError> {
    if root_feasibility.root_sha256() != root.root_sha256() {
        return Err(GenesisTrustStateAuthorizationError::RootFeasibilityRootMismatch);
    }
    if root_feasibility.principal_directory_sha256() != directory.directory_sha256() {
        return Err(
            GenesisTrustStateAuthorizationError::RootFeasibilityPrincipalDirectoryMismatch,
        );
    }

    let inner = authorize_genesis_trust_state_raw(
        root,
        directory,
        initial_snapshot,
        anchor,
        verifier,
    )
    .map_err(GenesisTrustStateAuthorizationError::Authorization)?;

    let authority_gate_sha256 = genesis_state_gate_digest(&inner, root_feasibility);
    Ok(AuthorizedTrustState {
        inner,
        root_signature_feasibility_sha256: root_feasibility.proof_sha256().clone(),
        authority_gate_sha256,
    })
}

pub fn authorize_role_under_root(
    state: &AuthorizedTrustState,
    proof: &RootRoleQuorumProof,
) -> Result<AuthorizedTrustRoleAttestation, AuthorizedRoleError> {
    let inner = authorize_role_under_root_raw(&state.inner, proof)?;
    let authority_sha256 = role_gate_digest(state, &inner);
    Ok(AuthorizedTrustRoleAttestation {
        inner,
        trust_state_gate_sha256: state.authority_gate_sha256().clone(),
        authority_sha256,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn authorize_root_transition(
    previous_state: &AuthorizedTrustState,
    previous_root: &FrozenTrustRoot,
    previous_root_feasibility: &TrustRootSignatureFeasibilityProof,
    next_root: &FrozenTrustRoot,
    next_root_feasibility: &TrustRootSignatureFeasibilityProof,
    next_snapshot: &TrustSnapshot,
    transition_at_unix_s: u64,
    contract: &TrustRootTransitionContract,
    old_root_proof: &RootRoleQuorumProof,
    new_root_proof: &RootRoleQuorumProof,
) -> Result<AuthorizedTrustState, RootTransitionTrustStateAuthorizationError> {
    if previous_root_feasibility.root_sha256() != previous_root.root_sha256() {
        return Err(
            RootTransitionTrustStateAuthorizationError::PreviousRootFeasibilityRootMismatch,
        );
    }
    if previous_root_feasibility.principal_directory_sha256()
        != previous_root.principal_directory_sha256()
    {
        return Err(
            RootTransitionTrustStateAuthorizationError::PreviousRootFeasibilityPrincipalDirectoryMismatch,
        );
    }
    if previous_root_feasibility.proof_sha256()
        != previous_state.root_signature_feasibility_sha256()
    {
        return Err(
            RootTransitionTrustStateAuthorizationError::PreviousRootFeasibilityProofMismatch,
        );
    }
    if next_root_feasibility.root_sha256() != next_root.root_sha256() {
        return Err(
            RootTransitionTrustStateAuthorizationError::NextRootFeasibilityRootMismatch,
        );
    }
    if next_root_feasibility.principal_directory_sha256()
        != next_root.principal_directory_sha256()
    {
        return Err(
            RootTransitionTrustStateAuthorizationError::NextRootFeasibilityPrincipalDirectoryMismatch,
        );
    }

    let inner = authorize_root_transition_raw(
        &previous_state.inner,
        previous_root,
        next_root,
        next_snapshot,
        transition_at_unix_s,
        contract,
        old_root_proof,
        new_root_proof,
    )
    .map_err(RootTransitionTrustStateAuthorizationError::Authorization)?;

    let authority_gate_sha256 = transition_state_gate_digest(
        previous_state,
        previous_root_feasibility,
        &inner,
        next_root_feasibility,
    );
    Ok(AuthorizedTrustState {
        inner,
        root_signature_feasibility_sha256: next_root_feasibility.proof_sha256().clone(),
        authority_gate_sha256,
    })
}

fn genesis_state_gate_digest(
    inner: &RawAuthorizedTrustState,
    feasibility: &TrustRootSignatureFeasibilityProof,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(TRUST_STATE_FEASIBILITY_GATE_DOMAIN);
    digest.text("genesis");
    digest.text(inner.root().authority_sha256().as_str());
    digest.text(inner.snapshot().authority_sha256().as_str());
    digest.text(feasibility.proof_sha256().as_str());
    digest.text("root-signature-feasibility-required");
    digest.digest()
}

fn transition_state_gate_digest(
    previous_state: &AuthorizedTrustState,
    previous_feasibility: &TrustRootSignatureFeasibilityProof,
    inner: &RawAuthorizedTrustState,
    next_feasibility: &TrustRootSignatureFeasibilityProof,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(TRUST_STATE_FEASIBILITY_GATE_DOMAIN);
    digest.text("transition");
    digest.text(previous_state.authority_gate_sha256().as_str());
    digest.text(previous_feasibility.proof_sha256().as_str());
    digest.text(inner.root().authority_sha256().as_str());
    digest.text(inner.snapshot().authority_sha256().as_str());
    digest.text(next_feasibility.proof_sha256().as_str());
    digest.text("root-signature-feasibility-required");
    digest.digest()
}

fn role_gate_digest(
    state: &AuthorizedTrustState,
    inner: &RawAuthorizedTrustRoleAttestation,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(TRUST_ROLE_FEASIBILITY_GATE_DOMAIN);
    digest.text(state.authority_gate_sha256().as_str());
    digest.text(inner.authority_sha256().as_str());
    digest.text(inner.role_quorum_proof_sha256().as_str());
    digest.text("root-signature-feasibility-transitively-required");
    digest.digest()
}
