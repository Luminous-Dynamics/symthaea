// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Registry-admitted topology discovery.
//!
//! This keeps topology on the same trust path as observations and identity:
//! the discoverer must already be registered, the request is validated, the
//! returned integration identity must match the registry slot, and topology
//! limits are applied independently of adapter code.

use crate::{
    DiscoveryRequest, DiscoverySnapshot, IntegrationError, IntegrationFuture, IntegrationId,
    IntegrationRegistry, TopologyLimits,
};

impl IntegrationRegistry {
    /// Validate an already-produced discovery snapshot against a registry slot
    /// and the conservative default topology budget.
    pub fn admit_discovery_snapshot(
        &self,
        id: &IntegrationId,
        snapshot: &DiscoverySnapshot,
    ) -> Result<(), IntegrationError> {
        self.admit_discovery_snapshot_with_limits(id, snapshot, &TopologyLimits::default())
    }

    /// Validate an already-produced discovery snapshot against an explicit
    /// centrally chosen budget. The adapter never controls this value.
    pub fn admit_discovery_snapshot_with_limits(
        &self,
        id: &IntegrationId,
        snapshot: &DiscoverySnapshot,
        limits: &TopologyLimits,
    ) -> Result<(), IntegrationError> {
        validate_discovery_for_limits(id, snapshot, limits)
    }

    /// Invoke a registered discoverer through the default topology admission
    /// boundary.
    pub fn discover<'a>(
        &'a self,
        id: &IntegrationId,
        request: DiscoveryRequest,
    ) -> IntegrationFuture<'a, Result<DiscoverySnapshot, IntegrationError>> {
        self.discover_with_limits(id, request, TopologyLimits::default())
    }

    /// Invoke a registered discoverer through an explicit topology budget.
    pub fn discover_with_limits<'a>(
        &'a self,
        id: &IntegrationId,
        request: DiscoveryRequest,
        limits: TopologyLimits,
    ) -> IntegrationFuture<'a, Result<DiscoverySnapshot, IntegrationError>> {
        let discoverer = self.discoverer(id).cloned();
        let integration_id = id.clone();

        Box::pin(async move {
            request.validate()?;
            let discoverer = discoverer.ok_or_else(|| {
                IntegrationError::Unsupported(format!(
                    "no discoverer registered for integration `{integration_id}`"
                ))
            })?;
            let snapshot = discoverer.discover(request).await?;
            validate_discovery_for_limits(&integration_id, &snapshot, &limits)?;
            Ok(snapshot)
        })
    }
}

fn validate_discovery_for_limits(
    id: &IntegrationId,
    snapshot: &DiscoverySnapshot,
    limits: &TopologyLimits,
) -> Result<(), IntegrationError> {
    if snapshot.integration_id != id.as_str() {
        return Err(IntegrationError::InvalidOutput(format!(
            "discoverer `{id}` returned snapshot attributed to `{}`",
            snapshot.integration_id
        )));
    }
    snapshot.validate_with_limits(limits).map_err(|error| {
        IntegrationError::InvalidOutput(format!(
            "integration `{id}` topology rejected by admission budget: {error}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AccessMode, CapabilityClass, CapabilityDeclaration, Discoverer, DiscoveredEntity,
        DiscoverySnapshot, EntityRef, INTEGRATION_MANIFEST_SCHEMA_VERSION, IntegrationIdentity,
        IntegrationManifest, MaturityLevel, RiskClass,
    };
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[derive(Clone)]
    struct FixtureDiscoverer {
        manifest: IntegrationManifest,
        integration_id_override: Option<String>,
        entity_count: usize,
    }

    impl IntegrationIdentity for FixtureDiscoverer {
        fn manifest(&self) -> &IntegrationManifest {
            &self.manifest
        }
    }

    impl Discoverer for FixtureDiscoverer {
        fn discover<'a>(
            &'a self,
            request: DiscoveryRequest,
        ) -> IntegrationFuture<'a, Result<DiscoverySnapshot, IntegrationError>> {
            let integration_id = self
                .integration_id_override
                .clone()
                .unwrap_or_else(|| self.manifest.id.0.clone());
            let entity_count = self.entity_count;
            Box::pin(async move {
                request.validate()?;
                Ok(DiscoverySnapshot {
                    integration_id,
                    discovered_at_unix_ms: 1,
                    entities: (0..entity_count)
                        .map(|index| DiscoveredEntity {
                            entity: EntityRef::new("fixture", "host", format!("node-{index}")),
                            display_name: None,
                            attributes: BTreeMap::new(),
                        })
                        .collect(),
                    relations: vec![],
                })
            })
        }
    }

    fn manifest() -> IntegrationManifest {
        IntegrationManifest {
            schema_version: INTEGRATION_MANIFEST_SCHEMA_VERSION,
            id: IntegrationId::new("fixture-topology"),
            display_name: "Fixture topology".into(),
            version: "0.1.0".into(),
            provider: "test".into(),
            protocols: vec!["fixture".into()],
            entity_kinds: vec!["host".into()],
            capabilities: vec![CapabilityDeclaration {
                name: "discover.fixture.topology".into(),
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

    #[test]
    fn source_identity_is_bound_to_registry_slot() {
        let integration = Arc::new(FixtureDiscoverer {
            manifest: manifest(),
            integration_id_override: Some("other-source".into()),
            entity_count: 1,
        });
        let mut registry = IntegrationRegistry::new();
        registry.register_discoverer(integration.clone()).unwrap();
        let snapshot = integration
            .discover(DiscoveryRequest::default());
        let _ = snapshot;

        let wrong = DiscoverySnapshot {
            integration_id: "other-source".into(),
            discovered_at_unix_ms: 1,
            entities: vec![],
            relations: vec![],
        };
        assert!(matches!(
            registry.admit_discovery_snapshot(&IntegrationId::new("fixture-topology"), &wrong),
            Err(IntegrationError::InvalidOutput(_))
        ));
    }

    #[test]
    fn central_entity_budget_rejects_oversized_snapshot() {
        let integration = Arc::new(FixtureDiscoverer {
            manifest: manifest(),
            integration_id_override: None,
            entity_count: 2,
        });
        let mut registry = IntegrationRegistry::new();
        registry.register_discoverer(integration).unwrap();

        let snapshot = DiscoverySnapshot {
            integration_id: "fixture-topology".into(),
            discovered_at_unix_ms: 1,
            entities: vec![
                DiscoveredEntity {
                    entity: EntityRef::new("fixture", "host", "a"),
                    display_name: None,
                    attributes: BTreeMap::new(),
                },
                DiscoveredEntity {
                    entity: EntityRef::new("fixture", "host", "b"),
                    display_name: None,
                    attributes: BTreeMap::new(),
                },
            ],
            relations: vec![],
        };
        let limits = TopologyLimits {
            max_entities: 1,
            ..Default::default()
        };
        assert!(matches!(
            registry.admit_discovery_snapshot_with_limits(
                &IntegrationId::new("fixture-topology"),
                &snapshot,
                &limits,
            ),
            Err(IntegrationError::InvalidOutput(_))
        ));
    }
}
