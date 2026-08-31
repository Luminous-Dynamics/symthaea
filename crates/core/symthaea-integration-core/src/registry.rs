// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability-aware registry for v0.1 read-only integrations.
//!
//! Registration is an admission boundary, not a passive lookup table. Every
//! adapter must pass the strict read-only manifest profile and must declare the
//! capability class matching the role under which it is registered.

use crate::manifest::{CapabilityClass, IntegrationId, IntegrationManifest, ManifestValidationError};
use crate::traits::{Discoverer, Observer};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Default)]
pub struct IntegrationRegistry {
    manifests: BTreeMap<IntegrationId, IntegrationManifest>,
    observers: BTreeMap<IntegrationId, Arc<dyn Observer>>,
    discoverers: BTreeMap<IntegrationId, Arc<dyn Discoverer>>,
}

impl IntegrationRegistry {
    pub fn new() -> Self {
        Self::default()
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

    pub fn manifest(&self, id: &IntegrationId) -> Option<&IntegrationManifest> {
        self.manifests.get(id)
    }

    pub fn observer(&self, id: &IntegrationId) -> Option<&Arc<dyn Observer>> {
        self.observers.get(id)
    }

    pub fn discoverer(&self, id: &IntegrationId) -> Option<&Arc<dyn Discoverer>> {
        self.discoverers.get(id)
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{
        AccessMode, CapabilityDeclaration, INTEGRATION_MANIFEST_SCHEMA_VERSION, MaturityLevel,
        RiskClass,
    };
    use crate::observation::ObservationBatch;
    use crate::topology::DiscoverySnapshot;
    use crate::traits::{
        DiscoveryRequest, IntegrationError, IntegrationFuture, IntegrationIdentity,
        ObservationRequest,
    };

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
            Box::pin(async move {
                Ok(ObservationBatch {
                    integration_id: id,
                    collected_at_unix_ms: 0,
                    observations: vec![],
                })
            })
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
    fn same_manifest_can_register_observe_and_discover_roles() {
        let integration = Arc::new(FixtureIntegration {
            manifest: manifest(&[CapabilityClass::Observe, CapabilityClass::Discover]),
        });
        let mut registry = IntegrationRegistry::new();
        registry.register_observer(integration.clone()).unwrap();
        registry.register_discoverer(integration).unwrap();
        assert_eq!(registry.integration_count(), 1);
        assert_eq!(registry.observer_count(), 1);
        assert_eq!(registry.discoverer_count(), 1);
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
            registry.register_discoverer(second),
            Err(RegistryError::ManifestCollision { .. })
        ));
    }
}
