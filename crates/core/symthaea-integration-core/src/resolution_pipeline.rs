// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Production-shaped identity resolution over registry-admitted provider snapshots.
//!
//! The lower-level resolver remains useful for tests and pure functions, but
//! infrastructure callers should prefer this pipeline: it refuses snapshots
//! from unregistered providers, reapplies the registry's identity admission
//! budget, prevents accidental duplicate-provider snapshots in one pass,
//! source-qualifies provider-local claim IDs, and then applies explicit
//! resolution work limits.

use crate::{
    EntityResolutionBatch, IdentitySnapshot, IntegrationId, IntegrationRegistry, ResolutionError,
    ResolutionLimits, normalize_kubernetes_uid_snapshot, resolve_identity_claims_with_limits,
};
use std::collections::BTreeSet;

/// Collision-safe reference used inside cross-provider resolution proposals.
///
/// Adapter claim IDs are only required to be unique inside one provider
/// snapshot. Qualifying them here prevents two vendors that both emit `claim-1`
/// from being treated as a duplicate global claim, while retaining a
/// deterministic mapping back to `(integration_id, local_claim_id)`.
pub fn source_qualified_claim_id(integration_id: &str, local_claim_id: &str) -> String {
    format!(
        "claim-ref-v1|{}:{integration_id}|{}:{local_claim_id}",
        integration_id.len(),
        local_claim_id.len()
    )
}

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

        for claim in &snapshot.claims {
            let mut claim = claim.clone();
            claim.claim_id = source_qualified_claim_id(
                &snapshot.integration_id,
                &claim.claim_id,
            );
            identity_claims.push(claim);
        }
        for claim in &snapshot.separation_claims {
            let mut claim = claim.clone();
            claim.claim_id = source_qualified_claim_id(
                &snapshot.integration_id,
                &claim.claim_id,
            );
            separation_claims.push(claim);
        }
    }

    resolve_identity_claims_with_limits(
        &identity_claims,
        &separation_claims,
        at_unix_ms,
        limits,
    )
    .map_err(ResolutionPipelineError::Resolution)
}

/// Explicit Kubernetes semantic-normalization + registry-admitted resolution.
///
/// This wrapper keeps Kubernetes/OTLP alias knowledge out of the resolver. Every
/// supplied snapshot is normalized through the shared semantic vocabulary using
/// a real Kubernetes cluster UID, then the ordinary registry admission and work
/// budgets are applied.
pub fn resolve_registry_kubernetes_uid_snapshots(
    registry: &IntegrationRegistry,
    snapshots: &[IdentitySnapshot],
    cluster_uid: &str,
    at_unix_ms: u64,
) -> Result<EntityResolutionBatch, ResolutionPipelineError> {
    resolve_registry_kubernetes_uid_snapshots_with_limits(
        registry,
        snapshots,
        cluster_uid,
        at_unix_ms,
        &ResolutionLimits::default(),
    )
}

pub fn resolve_registry_kubernetes_uid_snapshots_with_limits(
    registry: &IntegrationRegistry,
    snapshots: &[IdentitySnapshot],
    cluster_uid: &str,
    at_unix_ms: u64,
    limits: &ResolutionLimits,
) -> Result<EntityResolutionBatch, ResolutionPipelineError> {
    let mut normalized = Vec::with_capacity(snapshots.len());
    for snapshot in snapshots {
        normalized.push(
            normalize_kubernetes_uid_snapshot(snapshot, cluster_uid).map_err(|error| {
                ResolutionPipelineError::IdentityNormalizationRejected {
                    integration: snapshot.integration_id.clone(),
                    reason: error.to_string(),
                }
            })?,
        );
    }
    resolve_registry_identity_snapshots_with_limits(registry, &normalized, at_unix_ms, limits)
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ResolutionPipelineError {
    #[error("identity snapshot belongs to unregistered provider `{integration}`")]
    UnregisteredIdentityProvider { integration: String },
    #[error("identity provider `{integration}` supplied more than one snapshot in one resolution pass")]
    DuplicateProviderSnapshot { integration: String },
    #[error("identity snapshot from `{integration}` failed registry admission: {reason}")]
    AdmissionRejected { integration: String, reason: String },
    #[error("identity snapshot from `{integration}` failed semantic normalization: {reason}")]
    IdentityNormalizationRejected { integration: String, reason: String },
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
        IntegrationIdentity, IntegrationManifest, LEGACY_K8S_UID_SCHEME, MaturityLevel, RiskClass,
        ResolutionStatus,
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
                // Deliberately identical across providers. The pipeline must
                // source-qualify it before global resolution.
                claim_id: "claim-1".into(),
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

    fn k8s_snapshot(
        integration: &str,
        namespace: &str,
        scheme: &str,
        scope: Option<&str>,
    ) -> IdentitySnapshot {
        IdentitySnapshot {
            integration_id: integration.into(),
            collected_at_unix_ms: 100,
            claims: vec![IdentityClaim {
                claim_id: "pod-claim".into(),
                subject: EntityRef::new(namespace, "k8s_pod", format!("{integration}-pod")),
                identifier: ExternalIdentifier {
                    scheme: scheme.into(),
                    value: "pod-uid".into(),
                    scope: scope.map(str::to_string),
                    uniqueness: if scope.is_some() {
                        IdentifierUniqueness::Scoped
                    } else {
                        IdentifierUniqueness::Ambiguous
                    },
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

    #[test]
    fn source_qualified_claim_refs_are_unambiguous() {
        assert_ne!(
            source_qualified_claim_id("a", "b|c"),
            source_qualified_claim_id("a|b", "c")
        );
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
        let proposal = &resolved.proposals[0];
        assert_eq!(proposal.status, ResolutionStatus::StrongCandidateSame);
        assert_ne!(proposal.left, proposal.right);
        assert_eq!(proposal.identifier_matches.len(), 1);
        let matched = &proposal.identifier_matches[0];
        assert_eq!(matched.left_claim_ids.len(), 1);
        assert_eq!(matched.right_claim_ids.len(), 1);
        assert_ne!(matched.left_claim_ids[0], matched.right_claim_ids[0]);
        assert!(matched.left_claim_ids[0].starts_with("claim-ref-v1|"));
        assert!(matched.right_claim_ids[0].starts_with("claim-ref-v1|"));
    }

    #[test]
    fn kubernetes_pipeline_normalizes_legacy_and_standard_uid_claims_before_resolution() {
        let left_snapshot = k8s_snapshot(
            "kubernetes",
            "k8s-native",
            LEGACY_K8S_UID_SCHEME,
            Some("legacy-name-scope"),
        );
        let right_snapshot = k8s_snapshot(
            "otlp",
            "otel",
            "k8s.pod.uid",
            Some("old-otel-scope"),
        );
        let left = FixtureProvider {
            manifest: manifest("kubernetes"),
            snapshot: left_snapshot.clone(),
        };
        let right = FixtureProvider {
            manifest: manifest("otlp"),
            snapshot: right_snapshot.clone(),
        };
        let mut registry = IntegrationRegistry::new();
        registry.register_identity_provider(Arc::new(left)).unwrap();
        registry.register_identity_provider(Arc::new(right)).unwrap();

        let resolved = resolve_registry_kubernetes_uid_snapshots(
            &registry,
            &[left_snapshot, right_snapshot],
            "cluster-uid",
            100,
        )
        .unwrap();
        assert_eq!(resolved.proposals.len(), 1);
        assert_eq!(resolved.proposals[0].status, ResolutionStatus::StrongCandidateSame);
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
