// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact workload identity committed by Xenia agent authorizations.

use serde::{Deserialize, Serialize};
use symthaea_authority::{Digest32, PrincipalId};
use thiserror::Error;

const WORKLOAD_DOMAIN: &[u8] = b"symthaea.executor-workload.v1\0";

/// Security-relevant identity of the exact executor allowed to exercise a grant.
///
/// `artifact_digest` should be derived from independently measured executable
/// identity (for Nix deployments, preferably the qualified output/store identity
/// or a content commitment to it), not from a caller-provided version string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutorWorkloadV1 {
    /// Must equal the capability grant's exact audience/executor principal.
    pub executor: PrincipalId,
    /// Exact software/build artifact commitment.
    pub artifact_digest: Digest32,
    /// Exact security-relevant executor configuration commitment.
    pub configuration_digest: Digest32,
    /// Exact host/workload environment identity commitment.
    pub host_identity_digest: Digest32,
}

impl ExecutorWorkloadV1 {
    /// Validate non-placeholder workload identity inputs.
    pub fn validate(&self) -> Result<(), WorkloadIdentityError> {
        if self.executor.0.is_empty() {
            return Err(WorkloadIdentityError::EmptyExecutor);
        }
        if self.artifact_digest.0 == [0u8; 32] {
            return Err(WorkloadIdentityError::ZeroArtifactDigest);
        }
        if self.configuration_digest.0 == [0u8; 32] {
            return Err(WorkloadIdentityError::ZeroConfigurationDigest);
        }
        if self.host_identity_digest.0 == [0u8; 32] {
            return Err(WorkloadIdentityError::ZeroHostIdentityDigest);
        }
        Ok(())
    }

    /// Domain-separated deterministic commitment to the complete workload identity.
    pub fn digest(&self) -> Result<Digest32, WorkloadIdentityError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(WORKLOAD_DOMAIN);
        let executor = self.executor.0.as_bytes();
        hasher.update(&(executor.len() as u32).to_be_bytes());
        hasher.update(executor);
        hasher.update(&self.artifact_digest.0);
        hasher.update(&self.configuration_digest.0);
        hasher.update(&self.host_identity_digest.0);
        Ok(Digest32(*hasher.finalize().as_bytes()))
    }
}

/// Invalid workload measurement supplied to the authority verifier.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum WorkloadIdentityError {
    /// Executor identity cannot be empty.
    #[error("executor workload principal must not be empty")]
    EmptyExecutor,
    /// Artifact commitment cannot be a placeholder.
    #[error("executor artifact digest must not be zero")]
    ZeroArtifactDigest,
    /// Configuration commitment cannot be a placeholder.
    #[error("executor configuration digest must not be zero")]
    ZeroConfigurationDigest,
    /// Host identity commitment cannot be a placeholder.
    #[error("executor host identity digest must not be zero")]
    ZeroHostIdentityDigest,
}
