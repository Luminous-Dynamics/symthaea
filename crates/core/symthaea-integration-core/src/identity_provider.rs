// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generic read-only identity-evidence provider and admission budgets.
//!
//! Identity evidence is security-sensitive even when it cannot mutate upstream
//! systems: a buggy or malicious adapter could exhaust memory, impersonate a
//! different integration, or flood the resolver with weak aliases. This module
//! gives identity claims the same explicit admission boundary used by runtime
//! observations.

use crate::{
    EntityRef, IdentityClaim, IdentityValidationError, IntegrationError, IntegrationFuture,
    IntegrationIdentity, SeparationClaim,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct IdentityRequest {
    /// Empty means every entity visible inside the provider's configured scope.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entities: Vec<EntityRef>,
    /// Empty means every identifier scheme the provider emits.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub schemes: Vec<String>,
    /// Evaluate temporal claim validity at this instant when supplied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at_unix_ms: Option<u64>,
    /// Include explicit separation claims when true.
    pub include_separations: bool,
}

impl IdentityRequest {
    pub fn validate(&self) -> Result<(), IntegrationError> {
        if self.schemes.iter().any(|scheme| scheme.trim().is_empty()) {
            return Err(IntegrationError::InvalidRequest(
                "identity scheme selectors may not contain empty strings".into(),
            ));
        }
        Ok(())
    }
}

/// One provider result. The integration ID is checked again at registry
/// admission so an adapter cannot attribute claims to another registered source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IdentitySnapshot {
    pub integration_id: String,
    pub collected_at_unix_ms: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub claims: Vec<IdentityClaim>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub separation_claims: Vec<SeparationClaim>,
}

impl IdentitySnapshot {
    pub fn validate(&self) -> Result<(), IdentitySnapshotError> {
        if self.integration_id.trim().is_empty() {
            return Err(IdentitySnapshotError::EmptyIntegrationId);
        }

        let mut claim_ids = BTreeSet::new();
        for (index, claim) in self.claims.iter().enumerate() {
            claim
                .validate()
                .map_err(|reason| IdentitySnapshotError::InvalidIdentityClaim { index, reason })?;
            if claim.source.integration_id != self.integration_id {
                return Err(IdentitySnapshotError::IdentitySourceMismatch {
                    index,
                    snapshot: self.integration_id.clone(),
                    claim: claim.source.integration_id.clone(),
                });
            }
            if !claim_ids.insert(claim.claim_id.clone()) {
                return Err(IdentitySnapshotError::DuplicateClaimId(claim.claim_id.clone()));
            }
        }

        for (index, claim) in self.separation_claims.iter().enumerate() {
            claim.validate().map_err(|reason| {
                IdentitySnapshotError::InvalidSeparationClaim { index, reason }
            })?;
            if claim.source.integration_id != self.integration_id {
                return Err(IdentitySnapshotError::SeparationSourceMismatch {
                    index,
                    snapshot: self.integration_id.clone(),
                    claim: claim.source.integration_id.clone(),
                });
            }
            if !claim_ids.insert(claim.claim_id.clone()) {
                return Err(IdentitySnapshotError::DuplicateClaimId(claim.claim_id.clone()));
            }
        }
        Ok(())
    }

    pub fn validate_with_limits(
        &self,
        limits: &IdentityLimits,
    ) -> Result<(), IdentityAdmissionError> {
        self.validate()?;

        if self.claims.len() > limits.max_identity_claims {
            return Err(IdentityAdmissionError::TooManyIdentityClaims {
                actual: self.claims.len(),
                limit: limits.max_identity_claims,
            });
        }
        if self.separation_claims.len() > limits.max_separation_claims {
            return Err(IdentityAdmissionError::TooManySeparationClaims {
                actual: self.separation_claims.len(),
                limit: limits.max_separation_claims,
            });
        }

        let mut total_string_bytes = self.integration_id.len();
        for (index, claim) in self.claims.iter().enumerate() {
            validate_identifier_limits(index, claim, limits)?;
            if claim.evidence_observation_ids.len() > limits.max_evidence_refs_per_claim {
                return Err(IdentityAdmissionError::TooManyEvidenceReferences {
                    claim_id: claim.claim_id.clone(),
                    actual: claim.evidence_observation_ids.len(),
                    limit: limits.max_evidence_refs_per_claim,
                });
            }
            total_string_bytes = total_string_bytes
                .saturating_add(claim.claim_id.len())
                .saturating_add(claim.subject.namespace.len())
                .saturating_add(claim.subject.kind.len())
                .saturating_add(claim.subject.id.len())
                .saturating_add(claim.identifier.scheme.len())
                .saturating_add(claim.identifier.value.len())
                .saturating_add(claim.source.integration_id.len());
            if let Some(scope) = &claim.identifier.scope {
                total_string_bytes = total_string_bytes.saturating_add(scope.len());
            }
            if let Some(collector) = &claim.source.collector_id {
                total_string_bytes = total_string_bytes.saturating_add(collector.len());
            }
            if let Some(tenant) = &claim.source.tenant {
                total_string_bytes = total_string_bytes.saturating_add(tenant.len());
            }
            for evidence in &claim.evidence_observation_ids {
                total_string_bytes = total_string_bytes.saturating_add(evidence.0.len());
            }
            check_total_bytes(total_string_bytes, limits.max_total_string_bytes)?;
        }

        for claim in &self.separation_claims {
            if claim.evidence_observation_ids.len() > limits.max_evidence_refs_per_claim {
                return Err(IdentityAdmissionError::TooManyEvidenceReferences {
                    claim_id: claim.claim_id.clone(),
                    actual: claim.evidence_observation_ids.len(),
                    limit: limits.max_evidence_refs_per_claim,
                });
            }
            total_string_bytes = total_string_bytes
                .saturating_add(claim.claim_id.len())
                .saturating_add(claim.left.namespace.len())
                .saturating_add(claim.left.kind.len())
                .saturating_add(claim.left.id.len())
                .saturating_add(claim.right.namespace.len())
                .saturating_add(claim.right.kind.len())
                .saturating_add(claim.right.id.len())
                .saturating_add(claim.source.integration_id.len());
            if let Some(collector) = &claim.source.collector_id {
                total_string_bytes = total_string_bytes.saturating_add(collector.len());
            }
            if let Some(tenant) = &claim.source.tenant {
                total_string_bytes = total_string_bytes.saturating_add(tenant.len());
            }
            for evidence in &claim.evidence_observation_ids {
                total_string_bytes = total_string_bytes.saturating_add(evidence.0.len());
            }
            check_total_bytes(total_string_bytes, limits.max_total_string_bytes)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdentityLimits {
    pub max_identity_claims: usize,
    pub max_separation_claims: usize,
    pub max_evidence_refs_per_claim: usize,
    pub max_scheme_bytes: usize,
    pub max_identifier_value_bytes: usize,
    pub max_scope_bytes: usize,
    pub max_total_string_bytes: usize,
}

impl Default for IdentityLimits {
    fn default() -> Self {
        Self {
            max_identity_claims: 50_000,
            max_separation_claims: 10_000,
            max_evidence_refs_per_claim: 256,
            max_scheme_bytes: 256,
            max_identifier_value_bytes: 8_192,
            max_scope_bytes: 4_096,
            max_total_string_bytes: 16 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum IdentitySnapshotError {
    #[error("identity snapshot integration id is empty")]
    EmptyIntegrationId,
    #[error("identity claim {index} is invalid: {reason}")]
    InvalidIdentityClaim {
        index: usize,
        reason: IdentityValidationError,
    },
    #[error("separation claim {index} is invalid: {reason}")]
    InvalidSeparationClaim {
        index: usize,
        reason: IdentityValidationError,
    },
    #[error(
        "identity claim {index} belongs to integration `{claim}`, snapshot declares `{snapshot}`"
    )]
    IdentitySourceMismatch {
        index: usize,
        snapshot: String,
        claim: String,
    },
    #[error(
        "separation claim {index} belongs to integration `{claim}`, snapshot declares `{snapshot}`"
    )]
    SeparationSourceMismatch {
        index: usize,
        snapshot: String,
        claim: String,
    },
    #[error("duplicate identity/separation claim id `{0}`")]
    DuplicateClaimId(String),
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum IdentityAdmissionError {
    #[error("identity snapshot is structurally invalid: {0}")]
    InvalidSnapshot(#[from] IdentitySnapshotError),
    #[error("identity snapshot contains {actual} claims; limit is {limit}")]
    TooManyIdentityClaims { actual: usize, limit: usize },
    #[error("identity snapshot contains {actual} separation claims; limit is {limit}")]
    TooManySeparationClaims { actual: usize, limit: usize },
    #[error("identity claim {index} scheme is {actual} bytes; limit is {limit}")]
    SchemeTooLarge {
        index: usize,
        actual: usize,
        limit: usize,
    },
    #[error("identity claim {index} value is {actual} bytes; limit is {limit}")]
    IdentifierValueTooLarge {
        index: usize,
        actual: usize,
        limit: usize,
    },
    #[error("identity claim {index} scope is {actual} bytes; limit is {limit}")]
    ScopeTooLarge {
        index: usize,
        actual: usize,
        limit: usize,
    },
    #[error("claim `{claim_id}` has {actual} evidence references; limit is {limit}")]
    TooManyEvidenceReferences {
        claim_id: String,
        actual: usize,
        limit: usize,
    },
    #[error("identity snapshot string footprint exceeded {limit} bytes (at least {actual})")]
    TotalStringBytesExceeded { actual: usize, limit: usize },
}

pub trait IdentityProvider: IntegrationIdentity {
    fn identity_snapshot<'a>(
        &'a self,
        request: IdentityRequest,
    ) -> IntegrationFuture<'a, Result<IdentitySnapshot, IntegrationError>>;
}

fn validate_identifier_limits(
    index: usize,
    claim: &IdentityClaim,
    limits: &IdentityLimits,
) -> Result<(), IdentityAdmissionError> {
    if claim.identifier.scheme.len() > limits.max_scheme_bytes {
        return Err(IdentityAdmissionError::SchemeTooLarge {
            index,
            actual: claim.identifier.scheme.len(),
            limit: limits.max_scheme_bytes,
        });
    }
    if claim.identifier.value.len() > limits.max_identifier_value_bytes {
        return Err(IdentityAdmissionError::IdentifierValueTooLarge {
            index,
            actual: claim.identifier.value.len(),
            limit: limits.max_identifier_value_bytes,
        });
    }
    if let Some(scope) = &claim.identifier.scope {
        if scope.len() > limits.max_scope_bytes {
            return Err(IdentityAdmissionError::ScopeTooLarge {
                index,
                actual: scope.len(),
                limit: limits.max_scope_bytes,
            });
        }
    }
    Ok(())
}

fn check_total_bytes(actual: usize, limit: usize) -> Result<(), IdentityAdmissionError> {
    if actual > limit {
        Err(IdentityAdmissionError::TotalStringBytesExceeded { actual, limit })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ExternalIdentifier, IdentifierStability, IdentifierUniqueness, IdentityClaimSource,
        IdentityStrength,
    };

    fn claim(integration_id: &str) -> IdentityClaim {
        IdentityClaim {
            claim_id: "claim-1".into(),
            subject: EntityRef::new("test", "host", "node-1"),
            identifier: ExternalIdentifier {
                scheme: "host.id".into(),
                value: "uuid-1".into(),
                scope: None,
                uniqueness: IdentifierUniqueness::Global,
                stability: IdentifierStability::Persistent,
                case_sensitive: true,
            },
            strength: IdentityStrength::Strong,
            source_confidence: 1.0,
            source: IdentityClaimSource {
                integration_id: integration_id.into(),
                collector_id: None,
                tenant: None,
            },
            observed_at_unix_ms: 100,
            valid_from_unix_ms: None,
            valid_until_unix_ms: None,
            evidence_observation_ids: vec![],
        }
    }

    #[test]
    fn valid_snapshot_passes_default_limits() {
        let snapshot = IdentitySnapshot {
            integration_id: "fixture".into(),
            collected_at_unix_ms: 100,
            claims: vec![claim("fixture")],
            separation_claims: vec![],
        };
        assert!(snapshot.validate_with_limits(&IdentityLimits::default()).is_ok());
    }

    #[test]
    fn source_identity_smuggling_fails_structural_validation() {
        let snapshot = IdentitySnapshot {
            integration_id: "fixture".into(),
            collected_at_unix_ms: 100,
            claims: vec![claim("other")],
            separation_claims: vec![],
        };
        assert!(matches!(
            snapshot.validate(),
            Err(IdentitySnapshotError::IdentitySourceMismatch { .. })
        ));
    }

    #[test]
    fn evidence_reference_budget_is_enforced() {
        let mut identity = claim("fixture");
        identity.evidence_observation_ids = vec![
            crate::ObservationId::new("obs-1"),
            crate::ObservationId::new("obs-2"),
        ];
        let snapshot = IdentitySnapshot {
            integration_id: "fixture".into(),
            collected_at_unix_ms: 100,
            claims: vec![identity],
            separation_claims: vec![],
        };
        let limits = IdentityLimits {
            max_evidence_refs_per_claim: 1,
            ..IdentityLimits::default()
        };
        assert!(matches!(
            snapshot.validate_with_limits(&limits),
            Err(IdentityAdmissionError::TooManyEvidenceReferences { .. })
        ));
    }

    #[test]
    fn oversized_identifier_value_fails_closed() {
        let mut identity = claim("fixture");
        identity.identifier.value = "0123456789".into();
        let snapshot = IdentitySnapshot {
            integration_id: "fixture".into(),
            collected_at_unix_ms: 100,
            claims: vec![identity],
            separation_claims: vec![],
        };
        let limits = IdentityLimits {
            max_identifier_value_bytes: 8,
            ..IdentityLimits::default()
        };
        assert!(matches!(
            snapshot.validate_with_limits(&limits),
            Err(IdentityAdmissionError::IdentifierValueTooLarge { .. })
        ));
    }
}
