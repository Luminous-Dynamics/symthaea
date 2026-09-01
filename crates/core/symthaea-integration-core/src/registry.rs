// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability-aware registry for v0.1 read-only integrations.
//!
//! Registration is an admission boundary, not a passive lookup table. Every
//! adapter must pass the strict read-only manifest profile and must declare the
//! capability class matching the role under which it is registered. Runtime
//! observations and identity evidence also cross this boundary, where centrally
//! configured resource/cardinality budgets are enforced independently of
//! adapter code.

use crate::identity_provider::{
    IdentityLimits, IdentityProvider, IdentityRequest, IdentitySnapshot,
};
use crate::limits::ObservationLimits;
use crate::manifest::{
    CapabilityClass, IntegrationId, IntegrationManifest, ManifestValidationError,
};
use crate::observation::ObservationBatch;
use crate::traits::{
    Discoverer, IntegrationError, IntegrationFuture, ObservationRequest, Observer,
};
use std::collections::BTreeMap;
use std::sync::Arc;

pub struct IntegrationRegistry {
    manifests: BTreeMap<IntegrationId, IntegrationManifest>,
    observers: BTreeMap<IntegrationId, Arc<dyn Observer>>,
    discoverers: BTreeMap<IntegrationId, Arc<dyn Discoverer>>,
    identity_providers: BTreeMap<IntegrationId, Arc<dyn IdentityProvider>>,
    observation_limits: ObservationLimits,
    identity_limits: IdentityLimits,
}

impl Default for IntegrationRegistry {
    fn default() -> Self {
        Self {
            manifests: BTreeMap::new(),
            observers: BTreeMap::new(),
            discoverers: BTreeMap::new(),
            identity_providers: BTreeMap::new(),
            observation_limits: ObservationLimits::default(),
            identity_limits: IdentityLimits::default(),
        }
    }
}

impl IntegrationRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_observation_limits(observation_limits: ObservationLimits) -> Self {
        Self {
            observation_limits,
            ..Self::default()
        }
    }

    pub fn with_admission_limits(
        observation_limits: ObservationLimits,
        identity_limits: IdentityLimits,
    ) -> Self {
        Self {
            observation_limits,
            identity_limits,
            ..Self::default()
        }
    }

    pub fn observation_limits(&self) -> &ObservationLimits {
        &self.observation_limits
    }

    pub fn identity_limits(&self) -> &IdentityLimits {
        &self.identity_limits
    }

    /// Replace the central observation admission budget. Adapters cannot mutate it.
    pub fn set_observation_limits(&mut self, observation_limits: ObservationLimits) {
        self.observation_limits = observation_limits;
    }

    /// Replace the central identity-evidence admission budget. Adapters cannot mutate it.
    pub fn set_identity_limits(&mut self, identity_limits: IdentityLimits) {
        self.identity_limits = identity_limits;
    }

    pub fn register_observer(
        &mut self,
        observer: Arc<dyn Observer>,
    ) -> Result<(), RegistryError> {
        let manifest = observer.manifest().clone();
        self.admit_manifest(&manifest, CapabilityClass::Observe)?;
        let id = manifest.id.clone();

        if self.observers.contains_key(&id) {
            return Err(RegistryError::DuplicateRole {
                integration: id,
                role: CapabilityClass::Observe,
            });
        }

        self.manifests.entry(id.clone()).or_insert(manifest);
        self.observers.insert(id, observer);
        Ok(())
    }

    pub fn register_discoverer(
        &mut self,
        discoverer: Arc<dyn Discoverer>,
    ) -> Result<(), RegistryError> {
        let manifest = discoverer.manifest().clone();
        self.admit_manifest(&manifest, CapabilityClass::Discover)?;
        let id = manifest.id.clone();

        if self.discoverers.contains_key(&id) {
            return Err(RegistryError::DuplicateRole {
                integration: id,
                role: CapabilityClass::Discover,
            });
        }

        self.manifests.entry(id.clone()).or_insert(manifest);
        self.discoverers.insert(id, discoverer);
        Ok(())
    }

    pub fn register_identity_provider(
        &mut self,
        provider: Arc<dyn IdentityProvider>,
    ) -> Result<(), RegistryError> {
        let manifest = provider.manifest().clone();
        self.admit_manifest(&manifest, CapabilityClass::Discover)?;
        let id = manifest.id.clone();

        if self.identity_providers.contains_key(&id) {
            return Err(RegistryError::DuplicateIdentityProvider { integration: id });
        }

        self.manifests.entry(id.clone()).or_insert(manifest);
        self.identity_providers.insert(id, provider);
        Ok(())
    }

    /// Independently validate adapter observation output before world-model admission.
    pub fn admit_observation_batch(
        &self,
        id: &IntegrationId,
        batch: &ObservationBatch,
    ) -> Result<(), IntegrationError> {
        validate_batch_for_limits(id, batch, &self.observation_limits)
    }

    /// Independently validate identity evidence before resolution/world-model admission.
    pub fn admit_identity_snapshot(
        &self,
        id: &IntegrationId,
        snapshot: &IdentitySnapshot,
    ) -> Result<(), IntegrationError> {
        validate_identity_for_limits(id, snapshot, &self.identity_limits)
    }

    /// Invoke a registered observer through the central admission boundary.
    pub fn observe<'a>(
        &'a self,
        id: &IntegrationId,
        request: ObservationRequest,
    ) -> IntegrationFuture<'a, Result<ObservationBatch, IntegrationError>> {
        let observer = self.observers.get(id).cloned();
        let integration_id = id.clone();
        let limits = self.observation_limits.clone();

        Box::pin(async move {
            let observer = observer.ok_or_else(|| {
                IntegrationError::Unsupported(format!(
                    "no observer registered for integration `{integration_id}`"
                ))
            })?;
            let batch = observer.observe(request).await?;
            validate_batch_for_limits(&integration_id, &batch, &limits)?;
            Ok(batch)
        })
    }

    /// Invoke a registered identity provider through the central admission boundary.
    pub fn identity_snapshot<'a>(
        &'a self,
        id: &IntegrationId,
        request: IdentityRequest,
    ) -> IntegrationFuture<'a, Result<IdentitySnapshot, IntegrationError>> {
        let provider = self.identity_providers.get(id).cloned();
        let integration_id = id.clone();
        let limits = self.identity_limits.clone();

        Box::pin(async move {
            request.validate()?;
            let provider = provider.ok_or_else(|| {
                IntegrationError::Unsupported(format!(
                    "no identity provider registered for integration `{integration_id}`"
                ))
            })?;
            let snapshot = provider.identity_snapshot(request).await?;
            validate_identity_for_limits(&integration_id, &snapshot, &limits)?;
            Ok(snapshot)
        })
    }

    pub fn manifest(&self, id: &IntegrationId) -> Option<&IntegrationManifest> {
        self.manifests.get(id)
    }

    pub fn observer(&self, id: &IntegrationId) -> Option<&Arc<dyn Observer>> {
        self.observers.get(id)
    }

    pub fn discoverer(&self, id: &IntegrationId) -> Option<&Arc<dyn Discoverer>> {
        self.discoverers.get(id)
    }

    pub fn identity_provider(&self, id: &IntegrationId) -> Option<&Arc<dyn IdentityProvider>> {
        self.identity_providers.get(id)
    }

    pub fn manifests(&self) -> impl Iterator<Item = &IntegrationManifest> {
        self.manifests.values()
    }

    pub fn integration_count(&self) -> usize {
        self.manifests.len()
    }

    pub fn observer_count(&self) -> usize {
        self.observers.len()
    }

    pub fn discoverer_count(&self) -> usize {
        self.discoverers.len()
    }

    pub fn identity_provider_count(&self) -> usize {
        self.identity_providers.len()
    }

    fn admit_manifest(
        &self,
        manifest: &IntegrationManifest,
        required_role: CapabilityClass,
    ) -> Result<(), RegistryError> {
        manifest.validate_read_only_profile()?;

        if !manifest
            .capabilities
            .iter()
            .any(|capability| capability.class == required_role)
        {
            return Err(RegistryError::MissingCapabilityClass {
                integration: manifest.id.clone(),
                role: required_role,
            });
        }

        if let Some(existing) = self.manifests.get(&manifest.id) {
            if existing != manifest {
                return Err(RegistryError::ManifestCollision {
                    integration: manifest.id.clone(),
                });
            }
        }

        Ok(())
    }
}

fn validate_batch_for_limits(
    id: &IntegrationId,
    batch: &ObservationBatch,
    limits: &ObservationLimits,
) -> Result<(), IntegrationError> {
    if batch.integration_id != id.as_str() {
        return Err(IntegrationError::InvalidOutput(format!(
            "observer `{id}` returned batch attributed to `{}`",
            batch.integration_id
        )));
    }
    batch.validate_with_limits(limits).map_err(|error| {
        IntegrationError::InvalidOutput(format!(
            "integration `{id}` output rejected by observation admission budget: {error}"
        ))
    })
}

fn validate_identity_for_limits(
    id: &IntegrationId,
    snapshot: &IdentitySnapshot,
    limits: &IdentityLimits,
) -> Result<(), IntegrationError> {
    if snapshot.integration_id != id.as_str() {
        return Err(IntegrationError::InvalidOutput(format!(
            "identity provider `{id}` returned snapshot attributed to `{}`",
            snapshot.integration_id
        )));
    }
    snapshot.validate_with_limits(limits).map_err(|error| {
        IntegrationError::InvalidOutput(format!(
            "integration `{id}` identity output rejected by admission budget: {error}"
        ))
    })
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RegistryError {
    #[error("integration manifest rejected: {0}")]
    InvalidManifest(#[from] ManifestValidationError),
    #[error("integration `{integration}` does not declare required role {role:?}")]
    MissingCapabilityClass {
        integration: IntegrationId,
        role: CapabilityClass,
    },
    #[error("integration `{integration}` attempted to register conflicting manifests")]
    ManifestCollision { integration: IntegrationId },
    #[error("integration `{integration}` already has a registered {role:?} implementation")]
    DuplicateRole {
        integration: IntegrationId,
        role: CapabilityClass,
    },
    #[error("integration `{integration}` already has a registered identity provider")]
    DuplicateIdentityProvider { integration: IntegrationId },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{
        ExternalIdentifier, IdentifierStability, IdentifierUniqueness, IdentityClaim,
        IdentityClaimSource, IdentityStrength,
    };
    use crate::manifest::{
        AccessMode, CapabilityDeclaration, INTEGRATION_MANIFEST_SCHEMA_VERSION, MaturityLevel,
        RiskClass,
    };
    use crate::observation::{
        EntityRef, ObservationEnvelope, ObservationId, ObservationKind, ObservationLineage,
        ObservationQuality, ObservationSource, ObservationValue,
    };
    use crate::topology::DiscoverySnapshot;
    use crate::traits::{DiscoveryRequest, IntegrationIdentity};

    #[derive(Clone)]
    struct FixtureIntegration {
        manifest: IntegrationManifest,
    }

    impl IntegrationIdentity for FixtureIntegration {
        fn manifest(&self) -> &IntegrationManifest {
            &self.manifest
        }
    }

    impl Observer for FixtureIntegration {
        fn observe<'a>(
            &'a self,
            _request: ObservationRequest,
        ) -> IntegrationFuture<'a, Result<ObservationBatch, IntegrationError>> {
            let id = self.manifest.id.0.clone();
            Box::pin(async move { Ok(fixture_batch(&id)) })
        }
    }

    impl Discoverer for FixtureIntegration {
        fn discover<'a>(
            &'a self,
            _request: DiscoveryRequest,
        ) -> IntegrationFuture<'a, Result<DiscoverySnapshot, IntegrationError>> {
            let id = self.manifest.id.0.clone();
            Box::pin(async move {
                Ok(DiscoverySnapshot {
                    integration_id: id,
                    discovered_at_unix_ms: 0,
                    entities: vec![],
                    relations: vec![],
                })
            })
        }
    }

    impl IdentityProvider for FixtureIntegration {
        fn identity_snapshot<'a>(
            &'a self,
            request: IdentityRequest,
        ) -> IntegrationFuture<'a, Result<IdentitySnapshot, IntegrationError>> {
            let id = self.manifest.id.0.clone();
            Box::pin(async move {
                request.validate()?;
                Ok(fixture_identity_snapshot(&id))
            })
        }
    }

    fn fixture_batch(id: &str) -> ObservationBatch {
        ObservationBatch {
            integration_id: id.into(),
            collected_at_unix_ms: 2,
            observations: vec![ObservationEnvelope::new(
                ObservationId::new("fixture-observation"),
                1,
                2,
                EntityRef::new("test", "host", "node-1"),
                ObservationKind::Metric,
                "system.cpu.utilization",
                ObservationValue::Number {
                    value: 0.5,
                    unit: Some("1".into()),
                },
                ObservationSource {
                    integration_id: id.into(),
                    collector_id: None,
                    upstream_origin: None,
                    measurement_method: "fixture".into(),
                    tenant: None,
                },
                ObservationQuality::observed(1.0),
                ObservationLineage {
                    lineage_id: "fixture-lineage".into(),
                    parent_ids: vec![],
                    independence_group: None,
                    transforms: vec![],
                },
            )],
        }
    }

    fn fixture_identity_snapshot(id: &str) -> IdentitySnapshot {
        IdentitySnapshot {
            integration_id: id.into(),
            collected_at_unix_ms: 2,
            claims: vec![IdentityClaim {
                claim_id: "fixture-identity".into(),
                subject: EntityRef::new("test", "host", "node-1"),
                identifier: ExternalIdentifier {
                    scheme: "host.id".into(),
                    value: "uuid-node-1".into(),
                    scope: None,
                    uniqueness: IdentifierUniqueness::Global,
                    stability: IdentifierStability::Persistent,
                    case_sensitive: true,
                },
                strength: IdentityStrength::Strong,
                source_confidence: 1.0,
                source: IdentityClaimSource {
                    integration_id: id.into(),
                    collector_id: None,
                    tenant: None,
                },
                observed_at_unix_ms: 1,
                valid_from_unix_ms: None,
                valid_until_unix_ms: None,
                evidence_observation_ids: vec![ObservationId::new("fixture-observation")],
            }],
            separation_claims: vec![],
        }
    }

    fn manifest(classes: &[CapabilityClass]) -> IntegrationManifest {
        IntegrationManifest {
            schema_version: INTEGRATION_MANIFEST_SCHEMA_VERSION,
            id: IntegrationId::new("fixture"),
            display_name: "Fixture".into(),
            version: "0.1.0".into(),
            provider: "test".into(),
            protocols: vec!["fixture".into()],
            entity_kinds: vec!["host".into()],
            capabilities: classes
                .iter()
                .enumerate()
                .map(|(index, class)| CapabilityDeclaration {
                    name: format!("fixture.{index}"),
                    class: *class,
                    access: match class {
                        CapabilityClass::Observe | CapabilityClass::Discover => {
                            AccessMode::ReadOnly
                        }
                        CapabilityClass::Simulate => AccessMode::Simulated,
                        CapabilityClass::Actuate => AccessMode::Mutating,
                    },
                    risk: if *class == CapabilityClass::Actuate {
                        RiskClass::High
                    } else {
                        RiskClass::ReadOnly
                    },
                    reversible: false,
                    default_enabled: *class != CapabilityClass::Actuate,
                })
                .collect(),
            credentials: vec![],
            maturity: MaturityLevel::E1FixtureParsing,
            default_read_only: true,
        }
    }

    #[test]
    fn observer_requires_observe_capability() {
        let integration = Arc::new(FixtureIntegration {
            manifest: manifest(&[CapabilityClass::Discover]),
        });
        let mut registry = IntegrationRegistry::new();
        assert!(matches!(
            registry.register_observer(integration),
            Err(RegistryError::MissingCapabilityClass {
                role: CapabilityClass::Observe,
                ..
            })
        ));
    }

    #[test]
    fn identity_provider_requires_discover_capability() {
        let integration = Arc::new(FixtureIntegration {
            manifest: manifest(&[CapabilityClass::Observe]),
        });
        let mut registry = IntegrationRegistry::new();
        assert!(matches!(
            registry.register_identity_provider(integration),
            Err(RegistryError::MissingCapabilityClass {
                role: CapabilityClass::Discover,
                ..
            })
        ));
    }

    #[test]
    fn same_manifest_can_register_observe_discover_and_identity_roles() {
        let integration = Arc::new(FixtureIntegration {
            manifest: manifest(&[CapabilityClass::Observe, CapabilityClass::Discover]),
        });
        let mut registry = IntegrationRegistry::new();
        registry.register_observer(integration.clone()).unwrap();
        registry.register_discoverer(integration.clone()).unwrap();
        registry.register_identity_provider(integration).unwrap();
        assert_eq!(registry.integration_count(), 1);
        assert_eq!(registry.observer_count(), 1);
        assert_eq!(registry.discoverer_count(), 1);
        assert_eq!(registry.identity_provider_count(), 1);
    }

    #[test]
    fn actuation_manifest_is_rejected_by_v01_admission() {
        let integration = Arc::new(FixtureIntegration {
            manifest: manifest(&[CapabilityClass::Observe, CapabilityClass::Actuate]),
        });
        let mut registry = IntegrationRegistry::new();
        assert!(matches!(
            registry.register_observer(integration),
            Err(RegistryError::InvalidManifest(
                ManifestValidationError::ActuationDeclaredInReadOnlyProfile(_)
            ))
        ));
    }

    #[test]
    fn conflicting_manifest_for_same_id_is_rejected() {
        let first = Arc::new(FixtureIntegration {
            manifest: manifest(&[CapabilityClass::Observe, CapabilityClass::Discover]),
        });
        let mut second_manifest = manifest(&[CapabilityClass::Observe, CapabilityClass::Discover]);
        second_manifest.version = "0.2.0".into();
        let second = Arc::new(FixtureIntegration {
            manifest: second_manifest,
        });

        let mut registry = IntegrationRegistry::new();
        registry.register_observer(first).unwrap();
        assert!(matches!(
            registry.register_identity_provider(second),
            Err(RegistryError::ManifestCollision { .. })
        ));
    }

    #[test]
    fn registry_admission_enforces_central_batch_budget() {
        let registry = IntegrationRegistry::with_observation_limits(ObservationLimits {
            max_batch_observations: 0,
            ..ObservationLimits::default()
        });
        let result = registry.admit_observation_batch(
            &IntegrationId::new("fixture"),
            &fixture_batch("fixture"),
        );
        assert!(matches!(result, Err(IntegrationError::InvalidOutput(_))));
    }

    #[test]
    fn registry_admission_rejects_cross_integration_observation_identity_smuggling() {
        let registry = IntegrationRegistry::new();
        let result = registry.admit_observation_batch(
            &IntegrationId::new("fixture"),
            &fixture_batch("other-integration"),
        );
        assert!(matches!(result, Err(IntegrationError::InvalidOutput(_))));
    }

    #[test]
    fn registry_identity_admission_enforces_central_claim_budget() {
        let registry = IntegrationRegistry::with_admission_limits(
            ObservationLimits::default(),
            IdentityLimits {
                max_identity_claims: 0,
                ..IdentityLimits::default()
            },
        );
        let result = registry.admit_identity_snapshot(
            &IntegrationId::new("fixture"),
            &fixture_identity_snapshot("fixture"),
        );
        assert!(matches!(result, Err(IntegrationError::InvalidOutput(_))));
    }

    #[test]
    fn registry_identity_admission_rejects_cross_integration_smuggling() {
        let registry = IntegrationRegistry::new();
        let result = registry.admit_identity_snapshot(
            &IntegrationId::new("fixture"),
            &fixture_identity_snapshot("other-integration"),
        );
        assert!(matches!(result, Err(IntegrationError::InvalidOutput(_))));
    }
}
