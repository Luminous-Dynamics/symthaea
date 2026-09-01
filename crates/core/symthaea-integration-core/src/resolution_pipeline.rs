// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Production-shaped identity resolution over registry-admitted provider snapshots.
//!
//! The lower-level resolver remains useful for tests and pure functions, but
//! infrastructure callers should prefer this pipeline: it refuses snapshots
//! from unregistered providers, reapplies the registry's identity admission
//! budget, prevents accidental duplicate-provider snapshots in one pass, and
//! then applies explicit resolution work limits.

use crate::{
    EntityResolutionBatch, IdentitySnapshot, IntegrationId, IntegrationRegistry, ResolutionError,
    ResolutionLimits, resolve_identity_claims_with_limits,
};
use std::collections::BTreeSet;

pub fn resolve_registry_identity_snapshots(
    registry: &IntegrationRegistry,
    snapshots: &[IdentitySnapshot],
    at_unix_ms: u64,
) -> Result<EntityResolutionBatch, ResolutionPipelineError> {
    resolve_registry_identity_snapshots_with_limits(
        registry,
        snapshots,
        at_unix_ms,
        &ResolutionLimits::default(),
    )
}

pub fn resolve_registry_identity_snapshots_with_limits(
    registry: &IntegrationRegistry,
    snapshots: &[IdentitySnapshot],
    at_unix_ms: u64,
    limits: &ResolutionLimits,
) -> Result<EntityResolutionBatch, ResolutionPipelineError> {
    let mut seen_providers = BTreeSet::new();
    let mut identity_claims = Vec::new();
    let mut separation_claims = Vec::new();

    for snapshot in snapshots {
        let id = IntegrationId::new(snapshot.integration_id.clone());
        if registry.identity_provider(&id).is_none() {
            return Err(ResolutionPipelineError::UnregisteredIdentityProvider {
                integration: snapshot.integration_id.clone(),
            });
        }
        if !seen_providers.insert(snapshot.integration_id.clone()) {
            return Err(ResolutionPipelineError::DuplicateProviderSnapshot {
                integration: snapshot.integration_id.clone(),
            });
        }

        registry
            .admit_identity_snapshot(&id, snapshot)
            .map_err(|error| ResolutionPipelineError::AdmissionRejected {
                integration: snapshot.integration_id.clone(),
                reason: error.to_string(),
            })?;

        identity_claims.extend(snapshot.claims.iter().cloned());
        separation_claims.extend(snapshot.separation_claims.iter().cloned());
    }

    resolve_identity_claims_with_limits(
        &identity_claims,
        &separation_claims,
        at_unix_ms,
        limits,
    )
    .map_err(ResolutionPipelineError::Resolution)
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ResolutionPipelineError {
    #[error("identity snapshot belongs to unregistered provider `{integration}`")]
    UnregisteredIdentityProvider { integration: String },
    #[error("identity provider `{integration}` supplied more than one snapshot in one resolution pass")]
    DuplicateProviderSnapshot { integration: String },
    #[error("identity snapshot from `{integration}` failed registry admission: {reason}")]
    AdmissionRejected { integration: String, reason: String },
    #[error("identity resolution failed: {0}")]
    Resolution(#[from] ResolutionError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AccessMode, CapabilityClass, CapabilityDeclaration, EntityRef, ExternalIdentifier,
        IDENTITY_DISCOVERY_CAPABILITY, IdentifierStability, IdentifierUniqueness, IdentityClaim,
        IdentityClaimSource, IdentityProvider, IdentityRequest, IdentityStrength,
        INTEGRATION_MANIFEST_SCHEMA_VERSION, IntegrationError, IntegrationFuture,
        IntegrationIdentity, IntegrationManifest, MaturityLevel, RiskClass, ResolutionStatus,
    };
    use std::sync::Arc;

    #[derive(Clone)]
    struct FixtureProvider {
        manifest: IntegrationManifest,
        snapshot: IdentitySnapshot,
    }

    impl IntegrationIdentity for FixtureProvider {
        fn manifest(&self) -> &IntegrationManifest {
            &self.manifest
        }
    }

    impl IdentityProvider for FixtureProvider {
        fn identity_snapshot<'a>(
            &'a self,
            request: IdentityRequest,
        ) -> IntegrationFuture<'a, Result<IdentitySnapshot, IntegrationError>> {
            let snapshot = self.snapshot.clone();
            Box::pin(async move {
                request.validate()?;
                Ok(snapshot)
            })
        }
    }

    fn manifest(id: &str) -> IntegrationManifest {
        IntegrationManifest {
            schema_version: INTEGRATION_MANIFEST_SCHEMA_VERSION,
            id: IntegrationId::new(id),
            display_name: format!("Fixture {id}"),
            version: "0.1.0".into(),
            provider: "test".into(),
            protocols: vec!["fixture".into()],
            entity_kinds: vec!["host".into()],
            capabilities: vec![CapabilityDeclaration {
                name: IDENTITY_DISCOVERY_CAPABILITY.into(),
                class: CapabilityClass::Discover,
                access: AccessMode::ReadOnly,
                risk: RiskClass::ReadOnly,
                reversible: false,
                default_enabled: true,
            }],
            credentials: vec![],
            maturity: MaturityLevel::E1FixtureParsing,
            default_read_only: true,
        }
    }

    fn snapshot(integration: &str, namespace: &str, entity_id: &str) -> IdentitySnapshot {
        IdentitySnapshot {
            integration_id: integration.into(),
            collected_at_unix_ms: 100,
            claims: vec![IdentityClaim {
                claim_id: format!("{integration}:claim-1"),
                subject: EntityRef::new(namespace, "host", entity_id),
                identifier: ExternalIdentifier {
                    scheme: "host.id".into(),
                    value: "shared-uuid".into(),
                    scope: None,
                    uniqueness: IdentifierUniqueness::Global,
                    stability: IdentifierStability::Persistent,
                    case_sensitive: true,
                },
                strength: IdentityStrength::Strong,
                source_confidence: 1.0,
                source: IdentityClaimSource {
                    integration_id: integration.into(),
                    collector_id: None,
                    tenant: None,
                },
                observed_at_unix_ms: 100,
                valid_from_unix_ms: None,
                valid_until_unix_ms: None,
                evidence_observation_ids: vec![],
            }],
            separation_claims: vec![],
        }
    }

    fn provider(integration: &str, namespace: &str, entity_id: &str) -> FixtureProvider {
        FixtureProvider {
            manifest: manifest(integration),
            snapshot: snapshot(integration, namespace, entity_id),
        }
    }

    #[test]
    fn registered_admitted_snapshots_resolve_without_merging_local_refs() {
        let left = provider("left-source", "left", "node-a");
        let right = provider("right-source", "right", "node-b");
        let snapshots = vec![left.snapshot.clone(), right.snapshot.clone()];

        let mut registry = IntegrationRegistry::new();
        registry.register_identity_provider(Arc::new(left)).unwrap();
        registry.register_identity_provider(Arc::new(right)).unwrap();

        let resolved = resolve_registry_identity_snapshots(&registry, &snapshots, 100).unwrap();
        assert_eq!(resolved.proposals.len(), 1);
        assert_eq!(resolved.proposals[0].status, ResolutionStatus::StrongCandidateSame);
        assert_ne!(resolved.proposals[0].left, resolved.proposals[0].right);
    }

    #[test]
    fn unregistered_provider_snapshot_is_rejected_before_resolution() {
        let registry = IntegrationRegistry::new();
        let result = resolve_registry_identity_snapshots(
            &registry,
            &[snapshot("unregistered", "x", "node")],
            100,
        );
        assert!(matches!(
            result,
            Err(ResolutionPipelineError::UnregisteredIdentityProvider { .. })
        ));
    }

    #[test]
    fn duplicate_provider_snapshots_are_rejected_not_double_counted() {
        let provider = provider("source", "x", "node");
        let snapshot = provider.snapshot.clone();
        let mut registry = IntegrationRegistry::new();
        registry.register_identity_provider(Arc::new(provider)).unwrap();

        let result = resolve_registry_identity_snapshots(
            &registry,
            &[snapshot.clone(), snapshot],
            100,
        );
        assert!(matches!(
            result,
            Err(ResolutionPipelineError::DuplicateProviderSnapshot { .. })
        ));
    }
}
