// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! IF-12 local resource-scope coherence proof.
//!
//! Credential ids, quota scope ids, and provider account-scope ids are opaque
//! identifiers. Similar-looking strings do not prove that they refer to the same
//! provider account/deployment. This module verifies those relationships only
//! against an explicitly installed local authority registry.

#[cfg(not(test))]
use super::inference_binding::{CredentialStateBinding, QuotaStateBinding};
#[cfg(not(test))]
use super::inference_provider_registry::QualifiedProviderCandidate;

#[cfg(test)]
use crate::inference_binding::{CredentialStateBinding, QuotaStateBinding};
#[cfg(test)]
use crate::inference_provider_registry::QualifiedProviderCandidate;

use std::collections::BTreeMap;
use std::fmt;

const PROOF_DOMAIN: &[u8] = b"symthaea.inference.resource-scope-proof.v1";
const MAX_ID_BYTES: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct InferenceResourceScopeKey {
    provider_id: String,
    deployment_id: String,
    account_scope_id: String,
}

impl InferenceResourceScopeKey {
    pub fn new(
        provider_id: impl Into<String>,
        deployment_id: impl Into<String>,
        account_scope_id: impl Into<String>,
    ) -> Result<Self, InferenceResourceScopeError> {
        let provider_id = provider_id.into();
        let deployment_id = deployment_id.into();
        let account_scope_id = account_scope_id.into();
        validate_id(&provider_id).map_err(|_| InferenceResourceScopeError::InvalidProviderId)?;
        validate_id(&deployment_id)
            .map_err(|_| InferenceResourceScopeError::InvalidDeploymentId)?;
        validate_id(&account_scope_id)
            .map_err(|_| InferenceResourceScopeError::InvalidAccountScopeId)?;
        Ok(Self {
            provider_id,
            deployment_id,
            account_scope_id,
        })
    }

    pub fn provider_id(&self) -> &str {
        &self.provider_id
    }

    pub fn deployment_id(&self) -> &str {
        &self.deployment_id
    }

    pub fn account_scope_id(&self) -> &str {
        &self.account_scope_id
    }

    pub fn from_profile(
        profile: &QualifiedProviderCandidate,
    ) -> Result<Self, InferenceResourceScopeError> {
        Self::new(
            profile.candidate().provider_id.clone(),
            profile.deployment_id(),
            profile.account_scope_id(),
        )
    }
}

/// Evidence that a trusted local deployment/configuration authority installed one
/// resource mapping. The source payload itself is not retained here.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct LocalScopeRegistrationEvidence {
    authority_epoch: u64,
    source_digest: [u8; 32],
}

impl LocalScopeRegistrationEvidence {
    pub fn new(
        authority_epoch: u64,
        source_digest: [u8; 32],
    ) -> Result<Self, InferenceResourceScopeError> {
        if authority_epoch == 0 {
            return Err(InferenceResourceScopeError::ZeroAuthorityEpoch);
        }
        if source_digest == [0; 32] {
            return Err(InferenceResourceScopeError::ZeroSourceDigest);
        }
        Ok(Self {
            authority_epoch,
            source_digest,
        })
    }

    pub const fn authority_epoch(&self) -> u64 {
        self.authority_epoch
    }

    pub const fn source_digest(&self) -> &[u8; 32] {
        &self.source_digest
    }
}

impl fmt::Debug for LocalScopeRegistrationEvidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalScopeRegistrationEvidence")
            .field("authority_epoch", &self.authority_epoch)
            .field("source_digest", &self.source_digest)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CredentialIdentityKey {
    credential_id: String,
    epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct QuotaIdentityKey {
    scope_id: String,
    epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CredentialScopeRegistration {
    scope: InferenceResourceScopeKey,
    evidence: LocalScopeRegistrationEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QuotaScopeRegistration {
    scope: InferenceResourceScopeKey,
    evidence: LocalScopeRegistrationEvidence,
}

/// Locally authoritative resource-scope registry.
///
/// Mutable access to this registry is itself an authority boundary. IF-12 does not
/// claim that arbitrary untrusted code should be allowed to install registrations;
/// production wiring must place mutation behind local deployment/Xenia authority.
#[derive(Debug)]
pub struct InferenceResourceScopeRegistry {
    registry_id: [u8; 32],
    credentials: BTreeMap<CredentialIdentityKey, CredentialScopeRegistration>,
    quotas: BTreeMap<QuotaIdentityKey, QuotaScopeRegistration>,
    latest_credential_epoch: BTreeMap<String, u64>,
    latest_quota_epoch: BTreeMap<String, u64>,
}

impl InferenceResourceScopeRegistry {
    pub fn new(registry_id: [u8; 32]) -> Result<Self, InferenceResourceScopeError> {
        if registry_id == [0; 32] {
            return Err(InferenceResourceScopeError::ZeroRegistryId);
        }
        Ok(Self {
            registry_id,
            credentials: BTreeMap::new(),
            quotas: BTreeMap::new(),
            latest_credential_epoch: BTreeMap::new(),
            latest_quota_epoch: BTreeMap::new(),
        })
    }

    pub const fn registry_id(&self) -> &[u8; 32] {
        &self.registry_id
    }

    pub fn register_credential(
        &mut self,
        credential: &CredentialStateBinding,
        scope: InferenceResourceScopeKey,
        evidence: LocalScopeRegistrationEvidence,
    ) -> Result<(), InferenceResourceScopeError> {
        if self
            .latest_credential_epoch
            .get(credential.credential_id())
            .is_some_and(|latest| credential.epoch() <= *latest)
        {
            return Err(InferenceResourceScopeError::CredentialEpochNotAdvanced);
        }

        let key = CredentialIdentityKey {
            credential_id: credential.credential_id().to_owned(),
            epoch: credential.epoch(),
        };
        self.credentials.insert(
            key,
            CredentialScopeRegistration {
                scope,
                evidence,
            },
        );
        self.latest_credential_epoch
            .insert(credential.credential_id().to_owned(), credential.epoch());
        Ok(())
    }

    pub fn register_quota(
        &mut self,
        quota: &QuotaStateBinding,
        scope: InferenceResourceScopeKey,
        evidence: LocalScopeRegistrationEvidence,
    ) -> Result<(), InferenceResourceScopeError> {
        if self
            .latest_quota_epoch
            .get(quota.scope_id())
            .is_some_and(|latest| quota.epoch() <= *latest)
        {
            return Err(InferenceResourceScopeError::QuotaEpochNotAdvanced);
        }

        let key = QuotaIdentityKey {
            scope_id: quota.scope_id().to_owned(),
            epoch: quota.epoch(),
        };
        self.quotas.insert(
            key,
            QuotaScopeRegistration {
                scope,
                evidence,
            },
        );
        self.latest_quota_epoch
            .insert(quota.scope_id().to_owned(), quota.epoch());
        Ok(())
    }

    /// Verify that the current profile, credential identity/epoch, and quota
    /// identity/epoch are all explicitly registered to the same resource scope.
    pub fn verify(
        &self,
        profile: &QualifiedProviderCandidate,
        credential: &CredentialStateBinding,
        quota: &QuotaStateBinding,
    ) -> Result<VerifiedInferenceResourceScope, InferenceResourceScopeError> {
        let profile_scope = InferenceResourceScopeKey::from_profile(profile)?;

        let latest_credential = self
            .latest_credential_epoch
            .get(credential.credential_id())
            .ok_or(InferenceResourceScopeError::CredentialRegistrationMissing)?;
        if credential.epoch() != *latest_credential {
            return Err(InferenceResourceScopeError::CredentialRegistrationSuperseded);
        }
        let credential_key = CredentialIdentityKey {
            credential_id: credential.credential_id().to_owned(),
            epoch: credential.epoch(),
        };
        let credential_registration = self
            .credentials
            .get(&credential_key)
            .ok_or(InferenceResourceScopeError::CredentialRegistrationMissing)?;

        let latest_quota = self
            .latest_quota_epoch
            .get(quota.scope_id())
            .ok_or(InferenceResourceScopeError::QuotaRegistrationMissing)?;
        if quota.epoch() != *latest_quota {
            return Err(InferenceResourceScopeError::QuotaRegistrationSuperseded);
        }
        let quota_key = QuotaIdentityKey {
            scope_id: quota.scope_id().to_owned(),
            epoch: quota.epoch(),
        };
        let quota_registration = self
            .quotas
            .get(&quota_key)
            .ok_or(InferenceResourceScopeError::QuotaRegistrationMissing)?;

        if credential_registration.scope != profile_scope {
            return Err(InferenceResourceScopeError::CredentialScopeMismatch);
        }
        if quota_registration.scope != profile_scope {
            return Err(InferenceResourceScopeError::QuotaScopeMismatch);
        }

        let proof_digest = digest_scope_proof(
            self.registry_id,
            profile,
            credential,
            quota,
            credential_registration,
            quota_registration,
        );

        Ok(VerifiedInferenceResourceScope {
            scope: profile_scope,
            credential_id: credential.credential_id().to_owned(),
            credential_epoch: credential.epoch(),
            quota_scope_id: quota.scope_id().to_owned(),
            quota_epoch: quota.epoch(),
            proof_digest,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceScopeProofDigest([u8; 32]);

impl ResourceScopeProofDigest {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Debug for ResourceScopeProofDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ResourceScopeProofDigest({:02x}{:02x}{:02x}{:02x}…)",
            self.0[0], self.0[1], self.0[2], self.0[3]
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedInferenceResourceScope {
    scope: InferenceResourceScopeKey,
    credential_id: String,
    credential_epoch: u64,
    quota_scope_id: String,
    quota_epoch: u64,
    proof_digest: ResourceScopeProofDigest,
}

impl VerifiedInferenceResourceScope {
    pub fn scope(&self) -> &InferenceResourceScopeKey {
        &self.scope
    }

    pub fn credential_id(&self) -> &str {
        &self.credential_id
    }

    pub const fn credential_epoch(&self) -> u64 {
        self.credential_epoch
    }

    pub fn quota_scope_id(&self) -> &str {
        &self.quota_scope_id
    }

    pub const fn quota_epoch(&self) -> u64 {
        self.quota_epoch
    }

    pub const fn proof_digest(&self) -> ResourceScopeProofDigest {
        self.proof_digest
    }
}

fn digest_scope_proof(
    registry_id: [u8; 32],
    profile: &QualifiedProviderCandidate,
    credential: &CredentialStateBinding,
    quota: &QuotaStateBinding,
    credential_registration: &CredentialScopeRegistration,
    quota_registration: &QuotaScopeRegistration,
) -> ResourceScopeProofDigest {
    let mut hasher = blake3::Hasher::new();
    put_bytes(&mut hasher, PROOF_DOMAIN);
    put_bytes(&mut hasher, &registry_id);
    put_bytes(&mut hasher, profile.profile_digest().as_bytes());
    put_string(&mut hasher, profile.candidate().provider_id.as_str());
    put_string(&mut hasher, profile.deployment_id());
    put_string(&mut hasher, profile.account_scope_id());

    put_string(&mut hasher, credential.credential_id());
    hasher.update(&credential.epoch().to_le_bytes());
    hasher.update(
        &credential_registration
            .evidence
            .authority_epoch()
            .to_le_bytes(),
    );
    put_bytes(
        &mut hasher,
        credential_registration.evidence.source_digest(),
    );

    put_string(&mut hasher, quota.scope_id());
    hasher.update(&quota.epoch().to_le_bytes());
    hasher.update(&quota_registration.evidence.authority_epoch().to_le_bytes());
    put_bytes(&mut hasher, quota_registration.evidence.source_digest());

    ResourceScopeProofDigest(*hasher.finalize().as_bytes())
}

fn put_bytes(hasher: &mut blake3::Hasher, value: &[u8]) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn put_string(hasher: &mut blake3::Hasher, value: &str) {
    put_bytes(hasher, value.as_bytes());
}

fn validate_id(value: &str) -> Result<(), ()> {
    if value.trim().is_empty() || value.len() > MAX_ID_BYTES || value.as_bytes().contains(&0) {
        return Err(());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceResourceScopeError {
    InvalidProviderId,
    InvalidDeploymentId,
    InvalidAccountScopeId,
    ZeroAuthorityEpoch,
    ZeroSourceDigest,
    ZeroRegistryId,
    CredentialEpochNotAdvanced,
    QuotaEpochNotAdvanced,
    CredentialRegistrationMissing,
    QuotaRegistrationMissing,
    CredentialRegistrationSuperseded,
    QuotaRegistrationSuperseded,
    CredentialScopeMismatch,
    QuotaScopeMismatch,
}

impl fmt::Display for InferenceResourceScopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidProviderId => "resource scope provider id is invalid",
            Self::InvalidDeploymentId => "resource scope deployment id is invalid",
            Self::InvalidAccountScopeId => "resource scope account id is invalid",
            Self::ZeroAuthorityEpoch => "local scope authority epoch must be non-zero",
            Self::ZeroSourceDigest => "local scope source digest must be non-zero",
            Self::ZeroRegistryId => "resource scope registry id must be non-zero",
            Self::CredentialEpochNotAdvanced => "credential registration epoch did not advance",
            Self::QuotaEpochNotAdvanced => "quota registration epoch did not advance",
            Self::CredentialRegistrationMissing => "credential scope registration is missing",
            Self::QuotaRegistrationMissing => "quota scope registration is missing",
            Self::CredentialRegistrationSuperseded => "credential scope registration was superseded",
            Self::QuotaRegistrationSuperseded => "quota scope registration was superseded",
            Self::CredentialScopeMismatch => {
                "credential is not registered to the provider profile scope"
            }
            Self::QuotaScopeMismatch => "quota is not registered to the provider profile scope",
        };
        f.write_str(message)
    }
}

impl std::error::Error for InferenceResourceScopeError {}
