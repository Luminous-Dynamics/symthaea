// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Execution resource leasing above IF-7/IF-12/IF-13.
//!
//! A lease joins one conservative IF-7 reservation with one verified IF-12 scope
//! proof under a local authority lineage. It is non-Clone/non-Serde and remains
//! active across external I/O until terminal settlement or conservative abandon.

#[cfg(not(test))]
use super::inference_binding::{CredentialStateBinding, QuotaStateBinding};
#[cfg(not(test))]
use super::inference_executor::InferenceExecutionOutcome;
#[cfg(not(test))]
use super::inference_permit::{BindingDigest, InferenceExecutionBinding, InferencePermitError};
#[cfg(not(test))]
use super::inference_provider_registry::QualifiedProviderCandidate;
#[cfg(not(test))]
use super::inference_resource_guard::{
    InferenceResourceGuard, InferenceResourceGuardError, InferenceResourceReservation,
};
#[cfg(not(test))]
use super::inference_resource_scope::{
    InferenceResourceScopeError, InferenceResourceScopeRegistry, VerifiedInferenceResourceScope,
};

#[cfg(test)]
use crate::inference_binding::{CredentialStateBinding, QuotaStateBinding};
#[cfg(test)]
use crate::inference_executor::InferenceExecutionOutcome;
#[cfg(test)]
use crate::inference_permit::{BindingDigest, InferenceExecutionBinding, InferencePermitError};
#[cfg(test)]
use crate::inference_provider_registry::QualifiedProviderCandidate;
#[cfg(test)]
use crate::inference_resource_guard::{
    InferenceResourceGuard, InferenceResourceGuardError, InferenceResourceReservation,
};
#[cfg(test)]
use crate::inference_resource_scope::{
    InferenceResourceScopeError, InferenceResourceScopeRegistry, VerifiedInferenceResourceScope,
};

use std::collections::BTreeSet;
use std::fmt;

const V5_ROOT_DOMAIN: &[u8] = b"symthaea.inference.resource-lease-binding.v5";
const LEASE_DOMAIN: &[u8] = b"v4-provider-state-plus-active-resource-lease";
const LEASE_ID_DOMAIN: &[u8] = b"symthaea.inference.execution-resource-lease.v1";

/// Locally owned resource authority. The guard and scope registry are deliberately
/// private and no mutable references are exposed.
pub struct InferenceExecutionResourceAuthority {
    authority_id: [u8; 32],
    authority_epoch: u64,
    guard: InferenceResourceGuard,
    scopes: InferenceResourceScopeRegistry,
    next_lease_generation: u64,
    active_leases: BTreeSet<u64>,
}

impl fmt::Debug for InferenceExecutionResourceAuthority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InferenceExecutionResourceAuthority")
            .field("authority_epoch", &self.authority_epoch)
            .field("active_lease_count", &self.active_leases.len())
            .field("guard", &self.guard)
            .finish_non_exhaustive()
    }
}

impl InferenceExecutionResourceAuthority {
    pub fn new(
        authority_id: [u8; 32],
        guard: InferenceResourceGuard,
        scopes: InferenceResourceScopeRegistry,
    ) -> Result<Self, InferenceResourceLeaseError> {
        if authority_id == [0; 32] {
            return Err(InferenceResourceLeaseError::ZeroAuthorityId);
        }
        Ok(Self {
            authority_id,
            authority_epoch: 1,
            guard,
            scopes,
            next_lease_generation: 0,
            active_leases: BTreeSet::new(),
        })
    }

    pub const fn authority_epoch(&self) -> u64 {
        self.authority_epoch
    }

    pub fn active_lease_count(&self) -> usize {
        self.active_leases.len()
    }

    pub fn effective_remaining_requests(&self) -> Option<u64> {
        self.guard.effective_remaining_requests()
    }

    pub fn effective_remaining_tokens(&self) -> Option<u64> {
        self.guard.effective_remaining_tokens()
    }

    /// Cheap owner check used by the highest safe wrapper before it does provider
    /// profile resolution. This lets a capability presented to the wrong executor
    /// be returned intact instead of being orphaned by unrelated profile failure.
    pub(crate) fn validate_lease_owner(
        &self,
        lease: &InferenceExecutionResourceLease,
    ) -> Result<(), InferenceResourceLeaseError> {
        self.validate_lease_identity(lease)
    }

    pub fn verify_scope(
        &self,
        profile: &QualifiedProviderCandidate,
        credential: &CredentialStateBinding,
        quota: &QuotaStateBinding,
    ) -> Result<VerifiedInferenceResourceScope, InferenceResourceLeaseError> {
        self.ensure_guard_epochs(credential, quota)?;
        self.scopes
            .verify(profile, credential, quota)
            .map_err(InferenceResourceLeaseError::Scope)
    }

    /// Reserve local/provider resource authority after policy/profile/v4 admission
    /// has already succeeded. The proof is rechecked under this authority before
    /// the IF-7 reservation is minted.
    pub fn reserve_verified(
        &mut self,
        profile: &QualifiedProviderCandidate,
        credential: &CredentialStateBinding,
        quota: &QuotaStateBinding,
        verified_scope: &VerifiedInferenceResourceScope,
        reserved_tokens: u64,
        now_millis: u64,
    ) -> Result<InferenceExecutionResourceLease, InferenceResourceLeaseError> {
        self.ensure_guard_epochs(credential, quota)?;
        let current_scope = self
            .scopes
            .verify(profile, credential, quota)
            .map_err(InferenceResourceLeaseError::Scope)?;
        if current_scope.proof_digest() != verified_scope.proof_digest() {
            return Err(InferenceResourceLeaseError::ScopeChangedBeforeReservation);
        }

        let generation = self
            .next_lease_generation
            .checked_add(1)
            .ok_or(InferenceResourceLeaseError::LeaseGenerationOverflow)?;
        let lease_digest = digest_lease_identity(
            self.authority_id,
            self.authority_epoch,
            generation,
            reserved_tokens,
            verified_scope,
        )?;
        let reservation = self
            .guard
            .reserve(reserved_tokens, now_millis)
            .map_err(InferenceResourceLeaseError::Guard)?;

        self.next_lease_generation = generation;
        self.active_leases.insert(generation);
        Ok(InferenceExecutionResourceLease {
            authority_id: self.authority_id,
            authority_epoch: self.authority_epoch,
            generation,
            reserved_tokens,
            scope_proof: verified_scope.clone(),
            lease_digest,
            reservation,
        })
    }

    /// Re-prove that a lease still belongs to this exact authority and current
    /// scope state. An active lease intentionally survives changing numeric quota
    /// ceilings because its IF-7 reservation already carved out conservative
    /// request/token exposure before dispatch.
    pub fn revalidate(
        &self,
        lease: &InferenceExecutionResourceLease,
        profile: &QualifiedProviderCandidate,
        credential: &CredentialStateBinding,
        quota: &QuotaStateBinding,
    ) -> Result<VerifiedInferenceResourceScope, InferenceResourceLeaseError> {
        self.validate_lease_identity(lease)?;
        self.ensure_guard_epochs(credential, quota)?;
        let current_scope = self
            .scopes
            .verify(profile, credential, quota)
            .map_err(InferenceResourceLeaseError::Scope)?;
        if current_scope.proof_digest() != lease.scope_proof.proof_digest() {
            return Err(InferenceResourceLeaseError::ScopeChangedWhileLeased);
        }
        Ok(current_scope)
    }

    /// Settle a terminal inference outcome against the exact reservation held by
    /// the lease. HTTP status evidence differentiates 429 from 401/403 and other
    /// failures without retaining provider error bodies.
    pub fn settle_outcome(
        &mut self,
        lease: InferenceExecutionResourceLease,
        outcome: &InferenceExecutionOutcome,
        now_millis: u64,
    ) -> Result<InferenceResourceSettlement, InferenceResourceLeaseError> {
        self.validate_lease_identity(&lease)?;
        let generation = lease.generation;

        let settlement = if outcome.wire_observation.is_none() {
            self.guard
                .abandon(lease.reservation)
                .map_err(InferenceResourceLeaseError::Guard)?;
            InferenceResourceSettlement::AbandonedBeforeWire
        } else if outcome.failure.is_none() {
            if let Some(evidence) = outcome.receipt.wire_evidence() {
                self.guard
                    .settle_success(lease.reservation, evidence)
                    .map_err(InferenceResourceLeaseError::Guard)?;
                InferenceResourceSettlement::Success
            } else {
                self.guard
                    .settle_failure(lease.reservation, now_millis)
                    .map_err(InferenceResourceLeaseError::Guard)?;
                InferenceResourceSettlement::ConservativeFailure
            }
        } else {
            let evidence = outcome.receipt.wire_evidence();
            let status = evidence.and_then(|value| value.provider_http_status());
            match status {
                Some(429) => {
                    let evidence = evidence.ok_or(InferenceResourceLeaseError::MissingWireEvidence)?;
                    self.guard
                        .settle_rate_limited(lease.reservation, evidence, now_millis)
                        .map_err(InferenceResourceLeaseError::Guard)?;
                    InferenceResourceSettlement::RateLimited
                }
                Some(401) | Some(403) => {
                    self.guard
                        .settle_auth_rejected(lease.reservation)
                        .map_err(InferenceResourceLeaseError::Guard)?;
                    InferenceResourceSettlement::AuthRejected
                }
                _ => {
                    self.guard
                        .settle_failure(lease.reservation, now_millis)
                        .map_err(InferenceResourceLeaseError::Guard)?;
                    InferenceResourceSettlement::ConservativeFailure
                }
            }
        };

        if !self.active_leases.remove(&generation) {
            return Err(InferenceResourceLeaseError::InactiveLease);
        }
        Ok(settlement)
    }

    /// Retire a lease that provably will not reach a provider settlement path. IF-7
    /// remains conservative: reserved request/token budget is not refunded.
    pub fn abandon(
        &mut self,
        lease: InferenceExecutionResourceLease,
    ) -> Result<(), InferenceResourceLeaseError> {
        self.validate_lease_identity(&lease)?;
        let generation = lease.generation;
        self.guard
            .abandon(lease.reservation)
            .map_err(InferenceResourceLeaseError::Guard)?;
        if !self.active_leases.remove(&generation) {
            return Err(InferenceResourceLeaseError::InactiveLease);
        }
        Ok(())
    }

    /// Conservatively settle an executor-fatal path where no terminal outcome was
    /// available. The provider may have consumed work, so the reservation is not
    /// refunded.
    pub fn settle_executor_fatal(
        &mut self,
        lease: InferenceExecutionResourceLease,
        now_millis: u64,
    ) -> Result<(), InferenceResourceLeaseError> {
        self.validate_lease_identity(&lease)?;
        let generation = lease.generation;
        self.guard
            .settle_failure(lease.reservation, now_millis)
            .map_err(InferenceResourceLeaseError::Guard)?;
        if !self.active_leases.remove(&generation) {
            return Err(InferenceResourceLeaseError::InactiveLease);
        }
        Ok(())
    }

    /// Atomically replace the locally owned guard + scope-registry state only when
    /// no execution lease is active. This is the v1 mutation boundary; it avoids
    /// partially rotating one half of resource authority.
    pub fn replace_state(
        &mut self,
        guard: InferenceResourceGuard,
        scopes: InferenceResourceScopeRegistry,
    ) -> Result<(), InferenceResourceLeaseError> {
        if !self.active_leases.is_empty() {
            return Err(InferenceResourceLeaseError::ActiveLeasesPresent);
        }
        let next_epoch = self
            .authority_epoch
            .checked_add(1)
            .ok_or(InferenceResourceLeaseError::AuthorityEpochOverflow)?;
        self.guard = guard;
        self.scopes = scopes;
        self.authority_epoch = next_epoch;
        Ok(())
    }

    fn validate_lease_identity(
        &self,
        lease: &InferenceExecutionResourceLease,
    ) -> Result<(), InferenceResourceLeaseError> {
        if lease.authority_id != self.authority_id {
            return Err(InferenceResourceLeaseError::AuthorityMismatch);
        }
        if lease.authority_epoch != self.authority_epoch {
            return Err(InferenceResourceLeaseError::AuthorityEpochMismatch);
        }
        if !self.active_leases.contains(&lease.generation) {
            return Err(InferenceResourceLeaseError::InactiveLease);
        }
        Ok(())
    }

    fn ensure_guard_epochs(
        &self,
        credential: &CredentialStateBinding,
        quota: &QuotaStateBinding,
    ) -> Result<(), InferenceResourceLeaseError> {
        let guard_authority = self.guard.authority();
        if guard_authority.credential_epoch() != credential.epoch() {
            return Err(InferenceResourceLeaseError::GuardCredentialEpochMismatch);
        }
        if guard_authority.quota_epoch() != quota.epoch() {
            return Err(InferenceResourceLeaseError::GuardQuotaEpochMismatch);
        }
        Ok(())
    }
}

/// Non-clone execution resource capability. The opaque IF-7 reservation cannot be
/// separated from this lease by ordinary callers.
pub struct InferenceExecutionResourceLease {
    authority_id: [u8; 32],
    authority_epoch: u64,
    generation: u64,
    reserved_tokens: u64,
    scope_proof: VerifiedInferenceResourceScope,
    lease_digest: BindingDigest,
    reservation: InferenceResourceReservation,
}

impl fmt::Debug for InferenceExecutionResourceLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InferenceExecutionResourceLease")
            .field("authority_epoch", &self.authority_epoch)
            .field("generation", &self.generation)
            .field("reserved_tokens", &self.reserved_tokens)
            .field("lease_digest", &self.lease_digest)
            .finish_non_exhaustive()
    }
}

impl InferenceExecutionResourceLease {
    pub const fn authority_epoch(&self) -> u64 {
        self.authority_epoch
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn reserved_tokens(&self) -> u64 {
        self.reserved_tokens
    }

    pub const fn lease_digest(&self) -> BindingDigest {
        self.lease_digest
    }

    pub fn verified_scope(&self) -> &VerifiedInferenceResourceScope {
        &self.scope_proof
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceResourceSettlement {
    Success,
    RateLimited,
    AuthRejected,
    ConservativeFailure,
    AbandonedBeforeWire,
}

/// Compose one active lease into the already-qualified IF-13/v4 binding.
pub fn bind_active_resource_lease(
    mut v4_binding: InferenceExecutionBinding,
    lease: &InferenceExecutionResourceLease,
) -> Result<InferenceExecutionBinding, InferenceResourceLeaseError> {
    let mut hasher = blake3::Hasher::new();
    put_bytes(&mut hasher, V5_ROOT_DOMAIN);
    put_bytes(&mut hasher, LEASE_DOMAIN);
    hasher.update(v4_binding.provider_state_digest.as_bytes());
    hasher.update(lease.lease_digest.as_bytes());
    v4_binding.provider_state_digest = BindingDigest::new(*hasher.finalize().as_bytes())
        .map_err(InferenceResourceLeaseError::Permit)?;
    Ok(v4_binding)
}

fn digest_lease_identity(
    authority_id: [u8; 32],
    authority_epoch: u64,
    generation: u64,
    reserved_tokens: u64,
    scope: &VerifiedInferenceResourceScope,
) -> Result<BindingDigest, InferenceResourceLeaseError> {
    let mut hasher = blake3::Hasher::new();
    put_bytes(&mut hasher, LEASE_ID_DOMAIN);
    put_bytes(&mut hasher, &authority_id);
    hasher.update(&authority_epoch.to_le_bytes());
    hasher.update(&generation.to_le_bytes());
    hasher.update(&reserved_tokens.to_le_bytes());
    hasher.update(scope.proof_digest().as_bytes());
    BindingDigest::new(*hasher.finalize().as_bytes()).map_err(InferenceResourceLeaseError::Permit)
}

fn put_bytes(hasher: &mut blake3::Hasher, value: &[u8]) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

#[derive(Debug)]
pub enum InferenceResourceLeaseError {
    ZeroAuthorityId,
    LeaseGenerationOverflow,
    AuthorityEpochOverflow,
    AuthorityMismatch,
    AuthorityEpochMismatch,
    ActiveLeasesPresent,
    InactiveLease,
    GuardCredentialEpochMismatch,
    GuardQuotaEpochMismatch,
    ScopeChangedBeforeReservation,
    ScopeChangedWhileLeased,
    MissingWireEvidence,
    Guard(InferenceResourceGuardError),
    Scope(InferenceResourceScopeError),
    Permit(InferencePermitError),
}

impl fmt::Display for InferenceResourceLeaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroAuthorityId => write!(f, "resource lease authority id must be non-zero"),
            Self::LeaseGenerationOverflow => write!(f, "resource lease generation overflow"),
            Self::AuthorityEpochOverflow => write!(f, "resource authority epoch overflow"),
            Self::AuthorityMismatch => write!(f, "resource lease belongs to a different authority"),
            Self::AuthorityEpochMismatch => write!(f, "resource lease authority epoch changed"),
            Self::ActiveLeasesPresent => {
                write!(f, "active resource leases block authority replacement")
            }
            Self::InactiveLease => write!(f, "resource lease is inactive or already settled"),
            Self::GuardCredentialEpochMismatch => write!(
                f,
                "resource guard credential epoch does not match execution credential"
            ),
            Self::GuardQuotaEpochMismatch => {
                write!(f, "resource guard quota epoch does not match execution quota")
            }
            Self::ScopeChangedBeforeReservation => {
                write!(f, "resource scope changed before reservation")
            }
            Self::ScopeChangedWhileLeased => {
                write!(f, "resource scope changed while lease was active")
            }
            Self::MissingWireEvidence => {
                write!(f, "terminal settlement required missing wire evidence")
            }
            Self::Guard(error) => write!(f, "resource guard rejected lease operation: {error}"),
            Self::Scope(error) => write!(f, "resource scope rejected lease operation: {error}"),
            Self::Permit(error) => write!(f, "resource lease binding failed: {error}"),
        }
    }
}

impl std::error::Error for InferenceResourceLeaseError {}
