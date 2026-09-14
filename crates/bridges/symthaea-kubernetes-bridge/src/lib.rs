// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only Kubernetes API-object replay bridge.
//!
//! v0.1 is deliberately E1: callers supply already-decoded Kubernetes JSON
//! objects. This crate opens no API connection, reads no kubeconfig, accepts no
//! bearer token, and performs no mutation. It extracts a conservative topology
//! and UID-backed identity view suitable for the integration fabric.

#![forbid(unsafe_code)]

use blake3::Hasher;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_integration_core::{
    AccessMode, CapabilityClass, CapabilityDeclaration, Discoverer, DiscoveredEntity,
    DiscoveryRequest, DiscoverySnapshot, EntityRef, EntityRelation, ExternalIdentifier,
    IDENTITY_DISCOVERY_CAPABILITY, IdentifierStability, IdentifierUniqueness, IdentityClaim,
    IdentityClaimSource, IdentityProvider, IdentityRequest, IdentitySnapshot, IdentityStrength,
    INTEGRATION_MANIFEST_SCHEMA_VERSION, IntegrationError, IntegrationFuture, IntegrationId,
    IntegrationIdentity, IntegrationManifest, MaturityLevel, RelationBasis, RelationKind,
    RiskClass,
};

pub const KUBERNETES_INTEGRATION_ID: &str = "kubernetes-object-replay";
pub const KUBERNETES_DISCOVERY_CAPABILITY: &str = "discover.kubernetes.objects";

#[derive(Debug, Clone, PartialEq)]
pub struct KubernetesReplayContext {
    /// Stable caller-chosen cluster identity. This scopes Kubernetes UIDs and
    /// prevents two clusters from being merged merely because they reused the
    /// same textual UID/name in unrelated administrative domains.
    pub cluster_id: String,
    pub collector_id: Option<String>,
    pub tenant: Option<String>,
    pub source_confidence: f32,
}

impl Default for KubernetesReplayContext {
    fn default() -> Self {
        Self {
            cluster_id: "fixture-cluster".into(),
            collector_id: None,
            tenant: None,
            source_confidence: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KubernetesReplayDiscoverer {
    manifest: IntegrationManifest,
    context: KubernetesReplayContext,
    topology: DiscoverySnapshot,
    identity: IdentitySnapshot,
}

impl KubernetesReplayDiscoverer {
    pub fn from_objects(
        context: KubernetesReplayContext,
        documents: &[Value],
        collected_at_unix_ms: u64,
    ) -> Result<Self, IntegrationError> {
        validate_context(&context)?;
        let objects = expand_documents(documents)?;
        let mut parsed = Vec::with_capacity(objects.len());
        for (index, object) in objects.into_iter().enumerate() {
            parsed.push(parse_object(object, &context, index)?);
        }

        let cluster_entity = EntityRef::new(
            cluster_namespace(&context),
            "k8s_cluster",
            context.cluster_id.clone(),
        );
        let mut entities = BTreeMap::<String, DiscoveredEntity>::new();
        insert_entity(
            &mut entities,
            DiscoveredEntity {
                entity: cluster_entity.clone(),
                display_name: Some(context.cluster_id.clone()),
                attributes: BTreeMap::from([(
                    "k8s.cluster_id".into(),
                    context.cluster_id.clone(),
                )]),
            },
        )?;

        let mut uid_index = BTreeMap::<String, EntityRef>::new();
        let mut name_index = BTreeMap::<ObjectNameKey, EntityRef>::new();
        let mut identity_claims = BTreeMap::<String, IdentityClaim>::new();

        for object in &parsed {
            insert_entity(&mut entities, object.discovered.clone())?;
            let name_key = ObjectNameKey::new(&object.kind, object.namespace.as_deref(), &object.name);
            if let Some(existing) = name_index.insert(name_key, object.entity.clone()) {
                if existing != object.entity {
                    return Err(IntegrationError::InvalidOutput(format!(
                        "Kubernetes replay contains conflicting objects for kind/name `{}/{}`",
                        object.kind, object.name
                    )));
                }
            }
            if let Some(uid) = &object.uid {
                if let Some(existing) = uid_index.insert(uid.clone(), object.entity.clone()) {
                    if existing != object.entity {
                        return Err(IntegrationError::InvalidOutput(format!(
                            "Kubernetes UID `{uid}` is attached to multiple source-local entities"
                        )));
                    }
                }
                let claim = uid_identity_claim(
                    &context,
                    &object.entity,
                    uid,
                    collected_at_unix_ms,
                )?;
                identity_claims.insert(claim.claim_id.clone(), claim);
            }
        }

        let mut relation_keys = BTreeSet::<RelationKey>::new();
        let mut relations = Vec::new();

        // Every concrete Namespace and Node belongs to the cluster.
        for object in &parsed {
            if matches!(object.kind.as_str(), "Namespace" | "Node") {
                push_relation(
                    &mut relation_keys,
                    &mut relations,
                    EntityRelation {
                        from: object.entity.clone(),
                        to: cluster_entity.clone(),
                        kind: RelationKind::MemberOf,
                        basis: RelationBasis::Structural,
                        confidence: context.source_confidence,
                        observed_at_unix_ms: Some(collected_at_unix_ms),
                        evidence_observation_ids: vec![],
                        attributes: BTreeMap::from([(
                            "k8s.relationship".into(),
                            "cluster_membership".into(),
                        )]),
                    },
                );
            }
        }

        // Namespaced resources belong to their Namespace. Missing namespace
        // objects become explicit placeholders rather than dangling edges.
        for object in &parsed {
            if let Some(namespace_name) = object.namespace.as_deref() {
                let namespace_entity = resolve_or_placeholder(
                    &context,
                    "Namespace",
                    None,
                    namespace_name,
                    None,
                    &mut entities,
                    &mut uid_index,
                    &mut name_index,
                    &mut identity_claims,
                    collected_at_unix_ms,
                )?;
                push_relation(
                    &mut relation_keys,
                    &mut relations,
                    EntityRelation {
                        from: object.entity.clone(),
                        to: namespace_entity,
                        kind: RelationKind::MemberOf,
                        basis: RelationBasis::Structural,
                        confidence: context.source_confidence,
                        observed_at_unix_ms: Some(collected_at_unix_ms),
                        evidence_observation_ids: vec![],
                        attributes: BTreeMap::from([(
                            "k8s.relationship".into(),
                            "namespace_membership".into(),
                        )]),
                    },
                );
            }
        }

        // ownerReferences are structural ownership evidence from the API object.
        for object in &parsed {
            for owner in &object.owners {
                let owner_namespace = if is_cluster_scoped_kind(&owner.kind) {
                    None
                } else {
                    object.namespace.as_deref()
                };
                let owner_entity = resolve_or_placeholder(
                    &context,
                    &owner.kind,
                    owner_namespace,
                    &owner.name,
                    owner.uid.as_deref(),
                    &mut entities,
                    &mut uid_index,
                    &mut name_index,
                    &mut identity_claims,
                    collected_at_unix_ms,
                )?;
                let mut attributes = BTreeMap::from([(
                    "k8s.relationship".into(),
                    "owner_reference".into(),
                )]);
                if let Some(controller) = owner.controller {
                    attributes.insert("k8s.owner.controller".into(), controller.to_string());
                }
                if let Some(block) = owner.block_owner_deletion {
                    attributes.insert(
                        "k8s.owner.block_owner_deletion".into(),
                        block.to_string(),
                    );
                }
                push_relation(
                    &mut relation_keys,
                    &mut relations,
                    EntityRelation {
                        from: object.entity.clone(),
                        to: owner_entity,
                        kind: RelationKind::OwnedBy,
                        basis: RelationBasis::Structural,
                        confidence: context.source_confidence,
                        observed_at_unix_ms: Some(collected_at_unix_ms),
                        evidence_observation_ids: vec![],
                        attributes,
                    },
                );
            }
        }

        // Pod placement is explicit desired/observed scheduling topology.
        for object in &parsed {
            if object.kind == "Pod" {
                if let Some(node_name) = object.node_name.as_deref() {
                    let node = resolve_or_placeholder(
                        &context,
                        "Node",
                        None,
                        node_name,
                        None,
                        &mut entities,
                        &mut uid_index,
                        &mut name_index,
                        &mut identity_claims,
                        collected_at_unix_ms,
                    )?;
                    push_relation(
                        &mut relation_keys,
                        &mut relations,
                        EntityRelation {
                            from: object.entity.clone(),
                            to: node,
                            kind: RelationKind::HostedOn,
                            basis: RelationBasis::Structural,
                            confidence: context.source_confidence,
                            observed_at_unix_ms: Some(collected_at_unix_ms),
                            evidence_observation_ids: vec![],
                            attributes: BTreeMap::from([(
                                "k8s.relationship".into(),
                                "pod_node_assignment".into(),
                            )]),
                        },
                    );
                }
            }
        }

        // Service selectors are evaluated only against Pods present in this
        // replay corpus. Absence of a Pod therefore remains absence of evidence,
        // not evidence that the Service has no endpoint.
        let pods: Vec<&ParsedObject> = parsed.iter().filter(|object| object.kind == "Pod").collect();
        for service in parsed.iter().filter(|object| object.kind == "Service") {
            if service.service_selector.is_empty() {
                continue;
            }
            for pod in &pods {
                if pod.namespace != service.namespace {
                    continue;
                }
                if selector_matches(&service.service_selector, &pod.labels) {
                    push_relation(
                        &mut relation_keys,
                        &mut relations,
                        EntityRelation {
                            from: service.entity.clone(),
                            to: pod.entity.clone(),
                            kind: RelationKind::Serves,
                            basis: RelationBasis::Structural,
                            confidence: context.source_confidence,
                            observed_at_unix_ms: Some(collected_at_unix_ms),
                            evidence_observation_ids: vec![],
                            attributes: BTreeMap::from([(
                                "k8s.relationship".into(),
                                "service_selector_match".into(),
                            )]),
                        },
                    );
                }
            }
        }

        let topology = DiscoverySnapshot {
            integration_id: KUBERNETES_INTEGRATION_ID.into(),
            discovered_at_unix_ms: collected_at_unix_ms,
            entities: entities.into_values().collect(),
            relations,
        };
        topology
            .validate()
            .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;

        let identity = IdentitySnapshot {
            integration_id: KUBERNETES_INTEGRATION_ID.into(),
            collected_at_unix_ms,
            claims: identity_claims.into_values().collect(),
            separation_claims: vec![],
        };
        identity
            .validate()
            .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;

        Ok(Self {
            manifest: integration_manifest(),
            context,
            topology,
            identity,
        })
    }

    pub fn context(&self) -> &KubernetesReplayContext {
        &self.context
    }

    pub fn topology(&self) -> &DiscoverySnapshot {
        &self.topology
    }

    pub fn identity(&self) -> &IdentitySnapshot {
        &self.identity
    }

    pub fn discover_sync(
        &self,
        request: DiscoveryRequest,
    ) -> Result<DiscoverySnapshot, IntegrationError> {
        request.validate()?;
        let mut selected: BTreeSet<String> = self
            .topology
            .entities
            .iter()
            .filter(|entity| {
                request.entity_kinds.is_empty()
                    || request
                        .entity_kinds
                        .iter()
                        .any(|kind| kind == &entity.entity.kind)
            })
            .filter(|entity| {
                request
                    .filters
                    .iter()
                    .all(|(key, value)| entity.attributes.get(key) == Some(value))
            })
            .map(|entity| entity.entity.canonical_key())
            .collect();

        if let Some(root) = &request.root {
            let root_key = root.canonical_key();
            if selected.contains(&root_key) {
                let mut neighborhood = BTreeSet::from([root_key.clone()]);
                for relation in &self.topology.relations {
                    let from = relation.from.canonical_key();
                    let to = relation.to.canonical_key();
                    if from == root_key && selected.contains(&to) {
                        neighborhood.insert(to);
                    } else if to == root_key && selected.contains(&from) {
                        neighborhood.insert(from);
                    }
                }
                selected = neighborhood;
            } else {
                selected.clear();
            }
        }

        let entities = self
            .topology
            .entities
            .iter()
            .filter(|entity| selected.contains(&entity.entity.canonical_key()))
            .cloned()
            .collect::<Vec<_>>();
        let relations = self
            .topology
            .relations
            .iter()
            .filter(|relation| {
                selected.contains(&relation.from.canonical_key())
                    && selected.contains(&relation.to.canonical_key())
            })
            .cloned()
            .collect::<Vec<_>>();

        let snapshot = DiscoverySnapshot {
            integration_id: KUBERNETES_INTEGRATION_ID.into(),
            discovered_at_unix_ms: self.topology.discovered_at_unix_ms,
            entities,
            relations,
        };
        snapshot
            .validate()
            .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
        Ok(snapshot)
    }

    pub fn identity_snapshot_sync(
        &self,
        request: IdentityRequest,
    ) -> Result<IdentitySnapshot, IntegrationError> {
        request.validate()?;
        let claims = self
            .identity
            .claims
            .iter()
            .filter(|claim| {
                request.entities.is_empty()
                    || request.entities.iter().any(|entity| entity == &claim.subject)
            })
            .filter(|claim| {
                request.schemes.is_empty()
                    || request
                        .schemes
                        .iter()
                        .any(|scheme| scheme == &claim.identifier.scheme)
            })
            .filter(|claim| request.at_unix_ms.is_none_or(|at| claim.is_active_at(at)))
            .cloned()
            .collect();
        let snapshot = IdentitySnapshot {
            integration_id: KUBERNETES_INTEGRATION_ID.into(),
            collected_at_unix_ms: self.identity.collected_at_unix_ms,
            claims,
            separation_claims: vec![],
        };
        snapshot
            .validate()
            .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
        Ok(snapshot)
    }
}

impl IntegrationIdentity for KubernetesReplayDiscoverer {
    fn manifest(&self) -> &IntegrationManifest {
        &self.manifest
    }
}

impl Discoverer for KubernetesReplayDiscoverer {
    fn discover<'a>(
        &'a self,
        request: DiscoveryRequest,
    ) -> IntegrationFuture<'a, Result<DiscoverySnapshot, IntegrationError>> {
        Box::pin(async move { self.discover_sync(request) })
    }
}

impl IdentityProvider for KubernetesReplayDiscoverer {
    fn identity_snapshot<'a>(
        &'a self,
        request: IdentityRequest,
    ) -> IntegrationFuture<'a, Result<IdentitySnapshot, IntegrationError>> {
        Box::pin(async move { self.identity_snapshot_sync(request) })
    }
}

pub fn integration_manifest() -> IntegrationManifest {
    IntegrationManifest {
        schema_version: INTEGRATION_MANIFEST_SCHEMA_VERSION,
        id: IntegrationId::new(KUBERNETES_INTEGRATION_ID),
        display_name: "Kubernetes API Object Replay".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        provider: "Kubernetes / Symthaea".into(),
        protocols: vec!["kubernetes-api-object-json-replay".into()],
        entity_kinds: vec![
            "k8s_cluster".into(),
            "k8s_namespace".into(),
            "k8s_node".into(),
            "k8s_pod".into(),
            "k8s_service".into(),
            "k8s_deployment".into(),
            "k8s_replicaset".into(),
            "k8s_statefulset".into(),
            "k8s_daemonset".into(),
            "k8s_job".into(),
            "k8s_cronjob".into(),
            "k8s_configmap".into(),
            "k8s_secret".into(),
            "k8s_object".into(),
        ],
        capabilities: vec![
            CapabilityDeclaration {
                name: KUBERNETES_DISCOVERY_CAPABILITY.into(),
                class: CapabilityClass::Discover,
                access: AccessMode::ReadOnly,
                risk: RiskClass::ReadOnly,
                reversible: false,
                default_enabled: true,
            },
            CapabilityDeclaration {
                name: IDENTITY_DISCOVERY_CAPABILITY.into(),
                class: CapabilityClass::Discover,
                access: AccessMode::ReadOnly,
                risk: RiskClass::ReadOnly,
                reversible: false,
                default_enabled: true,
            },
        ],
        credentials: vec![],
        maturity: MaturityLevel::E1FixtureParsing,
        default_read_only: true,
    }
}

#[derive(Debug, Clone)]
struct ParsedObject {
    api_version: String,
    kind: String,
    name: String,
    namespace: Option<String>,
    uid: Option<String>,
    labels: BTreeMap<String, String>,
    owners: Vec<OwnerReference>,
    node_name: Option<String>,
    service_selector: BTreeMap<String, String>,
    entity: EntityRef,
    discovered: DiscoveredEntity,
}

#[derive(Debug, Clone)]
struct OwnerReference {
    kind: String,
    name: String,
    uid: Option<String>,
    controller: Option<bool>,
    block_owner_deletion: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ObjectNameKey {
    kind: String,
    namespace: Option<String>,
    name: String,
}

impl ObjectNameKey {
    fn new(kind: &str, namespace: Option<&str>, name: &str) -> Self {
        Self {
            kind: kind.into(),
            namespace: namespace.map(str::to_string),
            name: name.into(),
        }
    }
}

type RelationKey = (EntityRef, EntityRef, RelationKind, RelationBasis);

fn validate_context(context: &KubernetesReplayContext) -> Result<(), IntegrationError> {
    if context.cluster_id.trim().is_empty() {
        return Err(IntegrationError::InvalidRequest(
            "Kubernetes cluster_id may not be empty".into(),
        ));
    }
    if !context.source_confidence.is_finite()
        || !(0.0..=1.0).contains(&context.source_confidence)
    {
        return Err(IntegrationError::InvalidRequest(format!(
            "Kubernetes source_confidence must be within [0,1], got {}",
            context.source_confidence
        )));
    }
    Ok(())
}

fn expand_documents<'a>(documents: &'a [Value]) -> Result<Vec<&'a Value>, IntegrationError> {
    let mut queue = documents.iter().collect::<Vec<_>>();
    let mut output = Vec::new();
    while let Some(value) = queue.pop() {
        let kind = value.get("kind").and_then(Value::as_str).unwrap_or("");
        if kind.ends_with("List") {
            let items = value
                .get("items")
                .and_then(Value::as_array)
                .ok_or_else(|| IntegrationError::Protocol(format!(
                    "Kubernetes `{kind}` document has no array `items`"
                )))?;
            queue.extend(items.iter());
        } else {
            output.push(value);
        }
    }
    output.reverse();
    Ok(output)
}

fn parse_object(
    value: &Value,
    context: &KubernetesReplayContext,
    index: usize,
) -> Result<ParsedObject, IntegrationError> {
    let object = value.as_object().ok_or_else(|| {
        IntegrationError::Protocol(format!("Kubernetes document {index} is not a JSON object"))
    })?;
    let api_version = required_string(object.get("apiVersion"), index, "apiVersion")?;
    let kind = required_string(object.get("kind"), index, "kind")?;
    let metadata = object
        .get("metadata")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            IntegrationError::Protocol(format!(
                "Kubernetes document {index} has no object `metadata`"
            ))
        })?;
    let name = required_string(metadata.get("name"), index, "metadata.name")?;
    let namespace = optional_non_blank(metadata.get("namespace"));
    let uid = optional_non_blank(metadata.get("uid"));
    let labels = string_map(metadata.get("labels"), index, "metadata.labels")?;
    let owners = owner_references(metadata.get("ownerReferences"), index)?;
    let node_name = if kind == "Pod" {
        object
            .get("spec")
            .and_then(Value::as_object)
            .and_then(|spec| optional_non_blank(spec.get("nodeName")))
    } else {
        None
    };
    let service_selector = if kind == "Service" {
        object
            .get("spec")
            .and_then(Value::as_object)
            .map(|spec| string_map(spec.get("selector"), index, "spec.selector"))
            .transpose()?
            .unwrap_or_default()
    } else {
        BTreeMap::new()
    };

    let id = uid.clone().unwrap_or_else(|| {
        stable_name_id(context, &api_version, &kind, namespace.as_deref(), &name)
    });
    let entity = EntityRef::new(cluster_namespace(context), entity_kind(&kind), id);
    let mut attributes = BTreeMap::from([
        ("k8s.api_version".into(), api_version.clone()),
        ("k8s.kind".into(), kind.clone()),
        ("k8s.name".into(), name.clone()),
    ]);
    if let Some(namespace) = &namespace {
        attributes.insert("k8s.namespace".into(), namespace.clone());
    }
    if let Some(uid) = &uid {
        attributes.insert("k8s.uid".into(), uid.clone());
    } else {
        attributes.insert("k8s.identity_quality".into(), "name_fallback".into());
    }
    if let Some(resource_version) = optional_non_blank(metadata.get("resourceVersion")) {
        attributes.insert("k8s.resource_version".into(), resource_version);
    }
    if let Some(generation) = metadata.get("generation").and_then(Value::as_i64) {
        attributes.insert("k8s.generation".into(), generation.to_string());
    }
    for (key, value) in &labels {
        attributes.insert(format!("k8s.label.{key}"), value.clone());
    }
    // Deliberately do not import annotations or Secret data in E1. They often
    // contain large or sensitive material and are not required for identity or
    // the structural topology proved by this tranche.
    if kind == "Secret" {
        attributes.insert("k8s.payload_redacted".into(), "true".into());
    }

    Ok(ParsedObject {
        api_version,
        kind,
        name: name.clone(),
        namespace,
        uid,
        labels,
        owners,
        node_name,
        service_selector,
        entity: entity.clone(),
        discovered: DiscoveredEntity {
            entity,
            display_name: Some(name),
            attributes,
        },
    })
}

fn required_string(
    value: Option<&Value>,
    index: usize,
    field: &str,
) -> Result<String, IntegrationError> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            IntegrationError::Protocol(format!(
                "Kubernetes document {index} requires non-empty `{field}`"
            ))
        })
}

fn optional_non_blank(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn string_map(
    value: Option<&Value>,
    index: usize,
    field: &str,
) -> Result<BTreeMap<String, String>, IntegrationError> {
    let Some(value) = value else {
        return Ok(BTreeMap::new());
    };
    if value.is_null() {
        return Ok(BTreeMap::new());
    }
    let map = value.as_object().ok_or_else(|| {
        IntegrationError::Protocol(format!(
            "Kubernetes document {index} `{field}` is not an object"
        ))
    })?;
    let mut output = BTreeMap::new();
    for (key, value) in map {
        let string = value.as_str().ok_or_else(|| {
            IntegrationError::Protocol(format!(
                "Kubernetes document {index} `{field}.{key}` is not a string"
            ))
        })?;
        output.insert(key.clone(), string.to_string());
    }
    Ok(output)
}

fn owner_references(
    value: Option<&Value>,
    index: usize,
) -> Result<Vec<OwnerReference>, IntegrationError> {
    let Some(value) = value else {
        return Ok(vec![]);
    };
    if value.is_null() {
        return Ok(vec![]);
    }
    let owners = value.as_array().ok_or_else(|| {
        IntegrationError::Protocol(format!(
            "Kubernetes document {index} `metadata.ownerReferences` is not an array"
        ))
    })?;
    owners
        .iter()
        .enumerate()
        .map(|(owner_index, owner)| {
            let owner = owner.as_object().ok_or_else(|| {
                IntegrationError::Protocol(format!(
                    "Kubernetes document {index} ownerReference {owner_index} is not an object"
                ))
            })?;
            Ok(OwnerReference {
                kind: required_string(owner.get("kind"), index, "ownerReferences.kind")?,
                name: required_string(owner.get("name"), index, "ownerReferences.name")?,
                uid: optional_non_blank(owner.get("uid")),
                controller: owner.get("controller").and_then(Value::as_bool),
                block_owner_deletion: owner
                    .get("blockOwnerDeletion")
                    .and_then(Value::as_bool),
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn resolve_or_placeholder(
    context: &KubernetesReplayContext,
    kind: &str,
    namespace: Option<&str>,
    name: &str,
    uid: Option<&str>,
    entities: &mut BTreeMap<String, DiscoveredEntity>,
    uid_index: &mut BTreeMap<String, EntityRef>,
    name_index: &mut BTreeMap<ObjectNameKey, EntityRef>,
    identity_claims: &mut BTreeMap<String, IdentityClaim>,
    collected_at_unix_ms: u64,
) -> Result<EntityRef, IntegrationError> {
    if let Some(uid) = uid {
        if let Some(entity) = uid_index.get(uid) {
            return Ok(entity.clone());
        }
    }
    let name_key = ObjectNameKey::new(kind, namespace, name);
    if let Some(entity) = name_index.get(&name_key) {
        return Ok(entity.clone());
    }

    let id = uid
        .map(str::to_string)
        .unwrap_or_else(|| stable_name_id(context, "placeholder", kind, namespace, name));
    let entity = EntityRef::new(cluster_namespace(context), entity_kind(kind), id);
    let mut attributes = BTreeMap::from([
        ("k8s.kind".into(), kind.into()),
        ("k8s.name".into(), name.into()),
        ("k8s.placeholder".into(), "true".into()),
    ]);
    if let Some(namespace) = namespace {
        attributes.insert("k8s.namespace".into(), namespace.into());
    }
    if let Some(uid) = uid {
        attributes.insert("k8s.uid".into(), uid.into());
    }
    insert_entity(
        entities,
        DiscoveredEntity {
            entity: entity.clone(),
            display_name: Some(name.into()),
            attributes,
        },
    )?;
    name_index.insert(name_key, entity.clone());
    if let Some(uid) = uid {
        uid_index.insert(uid.into(), entity.clone());
        let claim = uid_identity_claim(context, &entity, uid, collected_at_unix_ms)?;
        identity_claims.entry(claim.claim_id.clone()).or_insert(claim);
    }
    Ok(entity)
}

fn insert_entity(
    entities: &mut BTreeMap<String, DiscoveredEntity>,
    entity: DiscoveredEntity,
) -> Result<(), IntegrationError> {
    let key = entity.entity.canonical_key();
    match entities.get(&key) {
        Some(existing) if existing != &entity => Err(IntegrationError::InvalidOutput(format!(
            "Kubernetes entity key collision for `{key}`"
        ))),
        Some(_) => Ok(()),
        None => {
            entities.insert(key, entity);
            Ok(())
        }
    }
}

fn push_relation(
    keys: &mut BTreeSet<RelationKey>,
    relations: &mut Vec<EntityRelation>,
    relation: EntityRelation,
) {
    let key = (
        relation.from.clone(),
        relation.to.clone(),
        relation.kind.clone(),
        relation.basis,
    );
    if keys.insert(key) {
        relations.push(relation);
    }
}

fn uid_identity_claim(
    context: &KubernetesReplayContext,
    subject: &EntityRef,
    uid: &str,
    observed_at_unix_ms: u64,
) -> Result<IdentityClaim, IntegrationError> {
    let identifier = ExternalIdentifier {
        scheme: "k8s.uid".into(),
        value: uid.into(),
        scope: Some(cluster_scope(context)),
        uniqueness: IdentifierUniqueness::Scoped,
        stability: IdentifierStability::Persistent,
        case_sensitive: true,
    };
    let identifier_key = identifier
        .canonical_key()
        .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
    let mut hasher = Hasher::new();
    feed_string(&mut hasher, KUBERNETES_INTEGRATION_ID);
    feed_string(&mut hasher, &subject.canonical_key());
    feed_string(&mut hasher, &identifier_key);
    let claim = IdentityClaim {
        claim_id: format!("k8s-identity:{}", hasher.finalize().to_hex()),
        subject: subject.clone(),
        identifier,
        strength: IdentityStrength::Strong,
        source_confidence: context.source_confidence,
        source: IdentityClaimSource {
            integration_id: KUBERNETES_INTEGRATION_ID.into(),
            collector_id: context.collector_id.clone(),
            tenant: context.tenant.clone(),
        },
        observed_at_unix_ms,
        valid_from_unix_ms: None,
        valid_until_unix_ms: None,
        evidence_observation_ids: vec![],
    };
    claim
        .validate()
        .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
    Ok(claim)
}

fn stable_name_id(
    context: &KubernetesReplayContext,
    api_version: &str,
    kind: &str,
    namespace: Option<&str>,
    name: &str,
) -> String {
    let mut hasher = Hasher::new();
    feed_string(&mut hasher, "symthaea-kubernetes-name-v1");
    feed_string(&mut hasher, &context.cluster_id);
    feed_string(&mut hasher, api_version);
    feed_string(&mut hasher, kind);
    feed_string(&mut hasher, namespace.unwrap_or(""));
    feed_string(&mut hasher, name);
    format!("name: {}", hasher.finalize().to_hex()).replace("name: ", "name:")
}

fn feed_string(hasher: &mut Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn cluster_namespace(context: &KubernetesReplayContext) -> String {
    format!("k8s:{}:{}", context.cluster_id.len(), context.cluster_id)
}

fn cluster_scope(context: &KubernetesReplayContext) -> String {
    format!("k8s.cluster:{}:{}", context.cluster_id.len(), context.cluster_id)
}

fn entity_kind(kind: &str) -> &'static str {
    match kind {
        "Namespace" => "k8s_namespace",
        "Node" => "k8s_node",
        "Pod" => "k8s_pod",
        "Service" => "k8s_service",
        "Deployment" => "k8s_deployment",
        "ReplicaSet" => "k8s_replicaset",
        "StatefulSet" => "k8s_statefulset",
        "DaemonSet" => "k8s_daemonset",
        "Job" => "k8s_job",
        "CronJob" => "k8s_cronjob",
        "ConfigMap" => "k8s_configmap",
        "Secret" => "k8s_secret",
        _ => "k8s_object",
    }
}

fn is_cluster_scoped_kind(kind: &str) -> bool {
    matches!(
        kind,
        "Namespace"
            | "Node"
            | "PersistentVolume"
            | "ClusterRole"
            | "ClusterRoleBinding"
            | "CustomResourceDefinition"
            | "StorageClass"
    )
}

fn selector_matches(
    selector: &BTreeMap<String, String>,
    labels: &BTreeMap<String, String>,
) -> bool {
    selector
        .iter()
        .all(|(key, value)| labels.get(key) == Some(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use symthaea_integration_core::{
        IntegrationRegistry, ResolutionStatus, resolve_registry_identity_snapshots,
    };
    use std::sync::Arc;

    fn fixture() -> KubernetesReplayDiscoverer {
        KubernetesReplayDiscoverer::from_objects(
            KubernetesReplayContext {
                cluster_id: "cluster-a".into(),
                source_confidence: 0.99,
                ..Default::default()
            },
            &[
                json!({
                    "apiVersion":"v1","kind":"Namespace",
                    "metadata":{"name":"shop","uid":"ns-1"}
                }),
                json!({
                    "apiVersion":"v1","kind":"Node",
                    "metadata":{"name":"node-a","uid":"node-1"}
                }),
                json!({
                    "apiVersion":"apps/v1","kind":"Deployment",
                    "metadata":{"name":"api","namespace":"shop","uid":"dep-1"}
                }),
                json!({
                    "apiVersion":"v1","kind":"Pod",
                    "metadata":{
                        "name":"api-pod","namespace":"shop","uid":"pod-1",
                        "labels":{"app":"api"},
                        "ownerReferences":[{
                            "apiVersion":"apps/v1","kind":"Deployment","name":"api",
                            "uid":"dep-1","controller":true
                        }]
                    },
                    "spec":{"nodeName":"node-a"}
                }),
                json!({
                    "apiVersion":"v1","kind":"Service",
                    "metadata":{"name":"api","namespace":"shop","uid":"svc-1"},
                    "spec":{"selector":{"app":"api"}}
                })
            ],
            100,
        )
        .unwrap()
    }

    #[test]
    fn manifest_is_read_only_e1_and_declares_identity_role() {
        let manifest = integration_manifest();
        assert_eq!(manifest.maturity, MaturityLevel::E1FixtureParsing);
        assert!(manifest.validate_read_only_profile().is_ok());
        assert!(manifest.declares(KUBERNETES_DISCOVERY_CAPABILITY));
        assert!(manifest.declares(IDENTITY_DISCOVERY_CAPABILITY));
    }

    #[test]
    fn fixture_builds_namespace_owner_node_and_service_topology() {
        let fixture = fixture();
        assert!(fixture.topology().validate().is_ok());
        let kinds = fixture
            .topology()
            .relations
            .iter()
            .map(|relation| relation.kind.clone())
            .collect::<BTreeSet<_>>();
        assert!(kinds.contains(&RelationKind::MemberOf));
        assert!(kinds.contains(&RelationKind::OwnedBy));
        assert!(kinds.contains(&RelationKind::HostedOn));
        assert!(kinds.contains(&RelationKind::Serves));
    }

    #[test]
    fn every_uid_becomes_cluster_scoped_identity_evidence() {
        let fixture = fixture();
        assert_eq!(fixture.identity().claims.len(), 5);
        assert!(fixture.identity().claims.iter().all(|claim| {
            claim.identifier.scheme == "k8s.uid"
                && claim.identifier.uniqueness == IdentifierUniqueness::Scoped
                && claim.identifier.scope.as_deref() == Some("k8s.cluster:9:cluster-a")
        }));
    }

    #[test]
    fn missing_owner_becomes_placeholder_instead_of_dangling_edge() {
        let fixture = KubernetesReplayDiscoverer::from_objects(
            KubernetesReplayContext::default(),
            &[json!({
                "apiVersion":"v1","kind":"Pod",
                "metadata":{
                    "name":"pod","namespace":"shop","uid":"pod-1",
                    "ownerReferences":[{
                        "apiVersion":"apps/v1","kind":"ReplicaSet","name":"rs",
                        "uid":"rs-1"
                    }]
                }
            })],
            100,
        )
        .unwrap();
        assert!(fixture.topology().validate().is_ok());
        assert!(fixture.topology().entities.iter().any(|entity| {
            entity.attributes.get("k8s.placeholder") == Some(&"true".to_string())
        }));
        assert!(fixture.identity().claims.iter().any(|claim| claim.identifier.value == "rs-1"));
    }

    #[test]
    fn secret_payload_is_never_imported() {
        let fixture = KubernetesReplayDiscoverer::from_objects(
            KubernetesReplayContext::default(),
            &[json!({
                "apiVersion":"v1","kind":"Secret",
                "metadata":{"name":"db","namespace":"shop","uid":"secret-1"},
                "data":{"password":"c2VjcmV0"}
            })],
            100,
        )
        .unwrap();
        let secret = fixture
            .topology()
            .entities
            .iter()
            .find(|entity| entity.entity.kind == "k8s_secret")
            .unwrap();
        assert_eq!(
            secret.attributes.get("k8s.payload_redacted"),
            Some(&"true".to_string())
        );
        assert!(!secret.attributes.values().any(|value| value == "c2VjcmV0"));
    }

    #[test]
    fn registry_admits_kubernetes_topology_and_identity() {
        let fixture = Arc::new(fixture());
        let mut registry = IntegrationRegistry::new();
        registry.register_discoverer(fixture.clone()).unwrap();
        registry.register_identity_provider(fixture.clone()).unwrap();
        registry
            .admit_discovery_snapshot(
                &IntegrationId::new(KUBERNETES_INTEGRATION_ID),
                fixture.topology(),
            )
            .unwrap();
        registry
            .admit_identity_snapshot(
                &IntegrationId::new(KUBERNETES_INTEGRATION_ID),
                fixture.identity(),
            )
            .unwrap();
    }

    #[test]
    fn namespaced_uid_does_not_resolve_across_different_cluster_scopes() {
        let a = Arc::new(KubernetesReplayDiscoverer::from_objects(
            KubernetesReplayContext { cluster_id: "cluster-a".into(), ..Default::default() },
            &[json!({
                "apiVersion":"v1","kind":"Pod",
                "metadata":{"name":"pod","namespace":"shop","uid":"same-uid"}
            })],
            100,
        ).unwrap());
        let b = Arc::new(KubernetesReplayDiscoverer::from_objects(
            KubernetesReplayContext { cluster_id: "cluster-b".into(), ..Default::default() },
            &[json!({
                "apiVersion":"v1","kind":"Pod",
                "metadata":{"name":"pod","namespace":"shop","uid":"same-uid"}
            })],
            100,
        ).unwrap());

        // Two instances use the same manifest ID, so they cannot both be
        // registered simultaneously in one registry. Compare their raw claims
        // to prove the scope prevents a false identifier match.
        assert_ne!(a.identity().claims[0].identifier, b.identity().claims[0].identifier);
        let _ = ResolutionStatus::Indeterminate;
        let _ = resolve_registry_identity_snapshots;
    }
}
