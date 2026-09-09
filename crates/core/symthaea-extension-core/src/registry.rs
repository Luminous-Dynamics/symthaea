use crate::{CapabilityId, ExtensionId, ExtensionManifest, ManifestProblem};
use std::collections::{BTreeMap, BTreeSet};

/// Registration failure for the metadata-only extension catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    InvalidManifest(Vec<ManifestProblem>),
    DuplicateExtension(ExtensionId),
}

/// One unresolved capability dependency in the current registry snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsatisfiedRequirement {
    pub extension: ExtensionId,
    pub capability: CapabilityId,
}

/// Deterministic metadata registry for extension discovery.
///
/// This type intentionally owns no executable plugin objects and performs no
/// loading. Runtime hosts may keep native/WASM/remote handles elsewhere while
/// using this registry as the stable capability catalog.
#[derive(Debug, Clone, Default)]
pub struct ExtensionRegistry {
    extensions: BTreeMap<ExtensionId, ExtensionManifest>,
    capability_index: BTreeMap<CapabilityId, BTreeSet<ExtensionId>>,
}

impl ExtensionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a manifest and index all capabilities it provides.
    ///
    /// Registration validates structural invariants and rejects duplicate
    /// extension IDs rather than silently replacing an existing provider.
    pub fn register(&mut self, manifest: ExtensionManifest) -> Result<(), RegistryError> {
        manifest
            .validate()
            .map_err(RegistryError::InvalidManifest)?;

        if self.extensions.contains_key(&manifest.id) {
            return Err(RegistryError::DuplicateExtension(manifest.id));
        }

        let id = manifest.id.clone();
        for capability in &manifest.provides {
            self.capability_index
                .entry(capability.id.clone())
                .or_default()
                .insert(id.clone());
        }
        self.extensions.insert(id, manifest);
        Ok(())
    }

    /// Remove an extension and all of its capability-index entries.
    pub fn unregister(&mut self, id: &ExtensionId) -> Option<ExtensionManifest> {
        let manifest = self.extensions.remove(id)?;
        for capability in &manifest.provides {
            let remove_key = if let Some(providers) = self.capability_index.get_mut(&capability.id) {
                providers.remove(id);
                providers.is_empty()
            } else {
                false
            };
            if remove_key {
                self.capability_index.remove(&capability.id);
            }
        }
        Some(manifest)
    }

    pub fn get(&self, id: &ExtensionId) -> Option<&ExtensionManifest> {
        self.extensions.get(id)
    }

    pub fn contains(&self, id: &ExtensionId) -> bool {
        self.extensions.contains_key(id)
    }

    pub fn len(&self) -> usize {
        self.extensions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.extensions.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&ExtensionId, &ExtensionManifest)> {
        self.extensions.iter()
    }

    /// Return all providers for a semantic capability in deterministic ID order.
    pub fn providers_for(
        &self,
        capability: &CapabilityId,
    ) -> impl Iterator<Item = &ExtensionManifest> {
        self.capability_index
            .get(capability)
            .into_iter()
            .flat_map(|providers| providers.iter())
            .filter_map(|id| self.extensions.get(id))
    }

    pub fn has_provider(&self, capability: &CapabilityId) -> bool {
        self.capability_index
            .get(capability)
            .is_some_and(|providers| !providers.is_empty())
    }

    /// Report unresolved capability requirements for the whole catalog.
    ///
    /// This is intentionally separate from `register`: extensions may be
    /// discovered in any order, and a host can decide whether unresolved
    /// dependencies should block installation, activation, or only invocation.
    pub fn unsatisfied_requirements(&self) -> Vec<UnsatisfiedRequirement> {
        let mut missing = Vec::new();
        for (id, manifest) in &self.extensions {
            for capability in &manifest.requires {
                if !self.has_provider(capability) {
                    missing.push(UnsatisfiedRequirement {
                        extension: id.clone(),
                        capability: capability.clone(),
                    });
                }
            }
        }
        missing
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AbiVersion, CapabilityDescriptor, EffectClass, ExtensionKind, PermissionSet,
        ResourceBudget, RuntimeKind,
    };

    fn manifest(id: &str, provides: &[&str], requires: &[&str]) -> ExtensionManifest {
        ExtensionManifest {
            id: ExtensionId::new(id),
            name: id.into(),
            version: "1.0.0".into(),
            abi: AbiVersion::V1,
            kind: ExtensionKind::Domain,
            runtime: RuntimeKind::Wasm,
            description: String::new(),
            provides: provides
                .iter()
                .map(|capability| CapabilityDescriptor {
                    id: CapabilityId::new(*capability),
                    description: format!("provider for {capability}"),
                    effect: EffectClass::Pure,
                })
                .collect(),
            requires: requires
                .iter()
                .map(|capability| CapabilityId::new(*capability))
                .collect(),
            permissions: PermissionSet::default(),
            resources: ResourceBudget::default(),
        }
    }

    #[test]
    fn indexes_multiple_providers_without_registration_order_semantics() {
        let mut registry = ExtensionRegistry::new();
        registry
            .register(manifest(
                "org.example.fast",
                &["science.orbits.propagate"],
                &[],
            ))
            .unwrap();
        registry
            .register(manifest(
                "org.example.precise",
                &["science.orbits.propagate"],
                &[],
            ))
            .unwrap();

        let providers: Vec<_> = registry
            .providers_for(&CapabilityId::new("science.orbits.propagate"))
            .map(|manifest| manifest.id.as_str())
            .collect();
        assert_eq!(providers, vec!["org.example.fast", "org.example.precise"]);
    }

    #[test]
    fn duplicate_extension_ids_are_rejected() {
        let mut registry = ExtensionRegistry::new();
        let first = manifest("org.example.plugin", &["science.example.one"], &[]);
        registry.register(first.clone()).unwrap();
        assert_eq!(
            registry.register(first),
            Err(RegistryError::DuplicateExtension(ExtensionId::new(
                "org.example.plugin"
            )))
        );
    }

    #[test]
    fn unresolved_requirements_are_reported_after_discovery() {
        let mut registry = ExtensionRegistry::new();
        registry
            .register(manifest(
                "org.example.consumer",
                &["science.example.consumer"],
                &["science.example.source"],
            ))
            .unwrap();

        assert_eq!(
            registry.unsatisfied_requirements(),
            vec![UnsatisfiedRequirement {
                extension: ExtensionId::new("org.example.consumer"),
                capability: CapabilityId::new("science.example.source"),
            }]
        );

        registry
            .register(manifest(
                "org.example.source",
                &["science.example.source"],
                &[],
            ))
            .unwrap();
        assert!(registry.unsatisfied_requirements().is_empty());
    }

    #[test]
    fn unregister_cleans_capability_index() {
        let mut registry = ExtensionRegistry::new();
        let id = ExtensionId::new("org.example.plugin");
        let capability = CapabilityId::new("science.example.capability");
        registry
            .register(manifest(
                id.as_str(),
                &[capability.as_str()],
                &[],
            ))
            .unwrap();
        assert!(registry.has_provider(&capability));

        registry.unregister(&id).unwrap();
        assert!(!registry.has_provider(&capability));
        assert!(registry.is_empty());
    }
}
