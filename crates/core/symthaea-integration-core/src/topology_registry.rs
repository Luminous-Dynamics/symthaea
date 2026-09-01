// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Registry-admitted topology discovery.
//!
//! This keeps topology on the same trust path as observations and identity:
//! the discoverer must already be registered, the request is validated, the
//! returned integration identity must match the registry slot, declared entity
//! kinds are enforced, and topology limits are applied independently of adapter
//! code. Exhaustive/complete discovery is a separate named capability so
//! partial replay cannot silently turn missing objects into negative evidence.

use crate::{
    COMPLETE_DISCOVERY_CAPABILITY, DiscoveryRequest, DiscoverySnapshot, IntegrationError,
    IntegrationFuture, IntegrationId, IntegrationRegistry, TopologyLimits,
};
use std::collections::BTreeSet;

impl IntegrationRegistry {
    /// Validate a discovery request against the registered adapter's declared
    /// epistemic capability before invoking it.
    pub fn admit_discovery_request(
        &self,
        id: &IntegrationId,
        request: &DiscoveryRequest,
    ) -> Result<(), IntegrationError> {
        request.validate()?;
        let manifest = self.manifest(id).ok_or_else(|| {
            IntegrationError::Unsupported(format!(
                "no registered integration manifest for `{id}`"
            ))
        })?;
        if self.discoverer(id).is_none() {
            return Err(IntegrationError::Unsupported(format!(
                "no discoverer registered for integration `{id}`"
            )));
        }
        if request.require_complete && !manifest.declares(COMPLETE_DISCOVERY_CAPABILITY) {
            return Err(IntegrationError::Unsupported(format!(
                "integration `{id}` is not qualified for complete discovery; absence must remain unknown"
            )));
        }
        Ok(())
    }

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
        let manifest = self.manifest(id).ok_or_else(|| {
            IntegrationError::Unsupported(format!(
                "no registered integration manifest for topology source `{id}`"
            ))
        })?;
        validate_discovery_for_limits(id, snapshot, limits, &manifest.entity_kinds)
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
        let request_admission = self.admit_discovery_request(id, &request);
        let discoverer = self.discoverer(id).cloned();
        let integration_id = id.clone();
        let declared_entity_kinds = self
            .manifest(id)
            .map(|manifest| manifest.entity_kinds.clone())
            .unwrap_or_default();

        Box::pin(async move {
            request_admission?;
            let discoverer = discoverer.ok_or_else(|| {
                IntegrationError::Unsupported(format!(
                    "no discoverer registered for integration `{integration_id}`"
                ))
            })?;
            let snapshot = discoverer.discover(request).await?;
            validate_discovery_for_limits(
                &integration_id,
                &snapshot,
                &limits,
                &declared_entity_kinds,
            )?;
            Ok(snapshot)
        })
    }
}

fn validate_discovery_for_limits(
    id: &IntegrationId,
    snapshot: &DiscoverySnapshot,
    limits: &TopologyLimits,
    declared_entity_kinds: &[String],
) -> Result<(), IntegrationError> {
    if snapshot.integration_id != id.as_str() {
        return Err(IntegrationError::InvalidOutput(format!(
            "discoverer `{id}` returned snapshot attributed to `{}`",
            snapshot.integration_id
        )));
    }
    let declared = declared_entity_kinds
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    for entity in &snapshot.entities {
        if !declared.contains(entity.entity.kind.as_str()) {
            return Err(IntegrationError::InvalidOutput(format!(
                "integration `{id}` emitted undeclared topology entity kind `{}` for `{}`",
                entity.entity.kind,
                entity.entity.canonical_key()
            )));
        }
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

    fn manifest(complete: bool) -> IntegrationManifest {
        let mut capabilities = vec![CapabilityDeclaration {
            name: "discover.fixture.topology".into(),
            class: CapabilityClass::Discover,
            access: AccessMode::ReadOnly,
            risk: RiskClass::ReadOnly,
            reversible: false,
            default_enabled: true,
        }];
        if complete {
            capabilities.push(CapabilityDeclaration {
                name: COMPLETE_DISCOVERY_CAPABILITY.into(),
                class: CapabilityClass::Discover,
                access: AccessMode::ReadOnly,
                risk: RiskClass::ReadOnly,
                reversible: false,
                default_enabled: true,
            });
        }
        IntegrationManifest {
            schema_version: INTEGRATION_MANIFEST_SCHEMA_VERSION,
            id: IntegrationId::new("fixture-topology"),
            display_name: "Fixture topology".into(),
            version: "0.1.0".into(),
            provider: "test".into(),
            protocols: vec!["fixture".into()],
            entity_kinds: vec!["host".into()],
            capabilities,
            credentials: vec![],
            maturity: MaturityLevel::E1FixtureParsing,
            default_read_only: true,
        }
    }

    fn integration(complete: bool) -> Arc<FixtureDiscoverer> {
        Arc::new(FixtureDiscoverer {
            manifest: manifest(complete),
            integration_id_override: None,
            entity_count: 1,
        })
    }

    #[test]
    fn partial_discoverer_cannot_claim_complete_absence_semantics() {
        let mut registry = IntegrationRegistry::new();
        registry.register_discoverer(integration(false)).unwrap();
        let request = DiscoveryRequest {
            require_complete: true,
            ..Default::default()
        };
        assert!(matches!(
            registry.admit_discovery_request(&IntegrationId::new("fixture-topology"), &request),
            Err(IntegrationError::Unsupported(_))
        ));
    }

    #[test]
    fn explicitly_qualified_discoverer_may_accept_complete_request() {
        let mut registry = IntegrationRegistry::new();
        registry.register_discoverer(integration(true)).unwrap();
        let request = DiscoveryRequest {
            require_complete: true,
            ..Default::default()
        };
        assert!(registry
            .admit_discovery_request(&IntegrationId::new("fixture-topology"), &request)
            .is_ok());
    }

    #[test]
    fn source_identity_is_bound_to_registry_slot() {
        let integration = Arc::new(FixtureDiscoverer {
            manifest: manifest(false),
            integration_id_override: Some("other-source".into()),
            entity_count: 1,
        });
        let mut registry = IntegrationRegistry::new();
        registry.register_discoverer(integration).unwrap();

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
    fn undeclared_entity_kind_is_rejected() {
        let mut registry = IntegrationRegistry::new();
        registry.register_discoverer(integration(false)).unwrap();
        let snapshot = DiscoverySnapshot {
            integration_id: "fixture-topology".into(),
            discovered_at_unix_ms: 1,
            entities: vec![DiscoveredEntity {
                entity: EntityRef::new("fixture", "database", "db-1"),
                display_name: None,
                attributes: BTreeMap::new(),
            }],
            relations: vec![],
        };
        assert!(matches!(
            registry.admit_discovery_snapshot(&IntegrationId::new("fixture-topology"), &snapshot),
            Err(IntegrationError::InvalidOutput(_))
        ));
    }

    #[test]
    fn central_entity_budget_rejects_oversized_snapshot() {
        let mut registry = IntegrationRegistry::new();
        registry.register_discoverer(integration(false)).unwrap();

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
