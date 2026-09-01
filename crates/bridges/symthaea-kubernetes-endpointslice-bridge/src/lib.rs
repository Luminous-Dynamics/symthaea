// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only EndpointSlice topology augmentation for Kubernetes E1 replay.
//!
//! This crate is deliberately not a second Kubernetes discoverer. It consumes
//! the already-built `KubernetesReplayDiscoverer` graph and same-capture decoded
//! `discovery.k8s.io/v1` EndpointSlice JSON, then returns one augmented
//! `DiscoverySnapshot` attributed to the existing `kubernetes-object-replay`
//! integration. It opens no API connection and carries no credentials.

#![forbid(unsafe_code)]

use blake3::Hasher;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::net::IpAddr;
use symthaea_integration_core::{
    DiscoveredEntity, DiscoverySnapshot, EntityRef, EntityRelation, IntegrationError,
    RelationBasis, RelationKind,
};
use symthaea_kubernetes_bridge::{KUBERNETES_INTEGRATION_ID, KubernetesReplayDiscoverer};

pub const ENDPOINTSLICE_API_VERSION: &str = "discovery.k8s.io/v1";
pub const ENDPOINTSLICE_KIND: &str = "EndpointSlice";
pub const ENDPOINTSLICE_SERVICE_LABEL: &str = "kubernetes.io/service-name";
pub const ENDPOINTSLICE_MANAGED_BY_LABEL: &str = "endpointslice.kubernetes.io/managed-by";

const MAX_ENDPOINTS_PER_SLICE: usize = 1_000;
const MAX_PORTS_PER_SLICE: usize = 100;
const MAX_ADDRESSES_PER_ENDPOINT: usize = 100;
const MAX_HINTS_PER_DIMENSION: usize = 8;

type RelationKey = (EntityRef, EntityRef, RelationKind, RelationBasis);

/// Augment one Kubernetes replay snapshot with EndpointSlice routing topology.
///
/// E1 requires the base topology and EndpointSlice documents to come from the
/// same capture instant. Combining different capture times would create a
/// snapshot that falsely implies simultaneity.
pub fn augment_endpoint_slices(
    base: &KubernetesReplayDiscoverer,
    documents: &[Value],
    collected_at_unix_ms: u64,
) -> Result<DiscoverySnapshot, IntegrationError> {
    if base.topology().integration_id != KUBERNETES_INTEGRATION_ID {
        return Err(IntegrationError::InvalidRequest(format!(
            "EndpointSlice augmentation requires `{KUBERNETES_INTEGRATION_ID}` topology, got `{}`",
            base.topology().integration_id
        )));
    }
    if base.topology().discovered_at_unix_ms != collected_at_unix_ms {
        return Err(IntegrationError::InvalidRequest(format!(
            "EndpointSlice capture time {collected_at_unix_ms} does not match base Kubernetes capture time {}",
            base.topology().discovered_at_unix_ms
        )));
    }

    let cluster_namespace = base
        .topology()
        .entities
        .iter()
        .find(|entity| entity.entity.kind == "k8s_cluster")
        .map(|entity| entity.entity.namespace.clone())
        .ok_or_else(|| {
            IntegrationError::InvalidOutput(
                "base Kubernetes topology has no k8s_cluster entity".into(),
            )
        })?;

    let mut entities = base
        .topology()
        .entities
        .iter()
        .cloned()
        .map(|entity| (entity.entity.canonical_key(), entity))
        .collect::<BTreeMap<_, _>>();
    let mut relations = base
        .topology()
        .relations
        .iter()
        .cloned()
        .map(|relation| (relation_key(&relation), relation))
        .collect::<BTreeMap<_, _>>();

    for document in expand_documents(documents)? {
        let slice = parse_slice(document)?;
        let slice_entity = endpoint_slice_entity(
            base,
            &cluster_namespace,
            &slice,
            collected_at_unix_ms,
        );
        insert_entity(&mut entities, slice_entity.clone())?;

        if let Some(namespace) = find_entity_by_kind_name(
            base.topology(),
            "Namespace",
            None,
            &slice.namespace,
        ) {
            insert_relation(
                &mut relations,
                structural_relation(
                    slice_entity.entity.clone(),
                    namespace,
                    RelationKind::MemberOf,
                    collected_at_unix_ms,
                    "endpointslice_namespace_membership",
                    base.context().source_confidence,
                ),
            );
        }

        let service_entity = if let Some(service_name) = slice.service_name.as_deref() {
            Some(resolve_or_reference(
                base.topology(),
                &mut entities,
                &cluster_namespace,
                ReferenceSpec {
                    kind: "Service",
                    namespace: Some(&slice.namespace),
                    name: service_name,
                    uid: None,
                    reference_kind: "k8s_service_reference",
                },
            )?)
        } else {
            None
        };

        if let Some(service) = &service_entity {
            insert_relation(
                &mut relations,
                structural_relation(
                    slice_entity.entity.clone(),
                    service.clone(),
                    RelationKind::Other("EndpointSliceFor".into()),
                    collected_at_unix_ms,
                    "endpointslice_service_label",
                    base.context().source_confidence,
                ),
            );
        }

        for endpoint in &slice.endpoints {
            let endpoint_entity = endpoint_membership_entity(
                &cluster_namespace,
                &slice_entity.entity,
                &slice,
                endpoint,
            );
            insert_entity(&mut entities, endpoint_entity.clone())?;

            insert_relation(
                &mut relations,
                structural_relation(
                    endpoint_entity.entity.clone(),
                    slice_entity.entity.clone(),
                    RelationKind::MemberOf,
                    collected_at_unix_ms,
                    "endpoint_membership",
                    base.context().source_confidence,
                ),
            );

            if let Some(service) = &service_entity {
                insert_relation(
                    &mut relations,
                    structural_relation(
                        service.clone(),
                        endpoint_entity.entity.clone(),
                        RelationKind::Serves,
                        collected_at_unix_ms,
                        "service_endpointslice_membership",
                        base.context().source_confidence,
                    ),
                );
            }

            if let Some(target) = &endpoint.target_ref {
                let target_entity = resolve_or_reference(
                    base.topology(),
                    &mut entities,
                    &cluster_namespace,
                    ReferenceSpec {
                        kind: &target.kind,
                        namespace: target.namespace.as_deref(),
                        name: &target.name,
                        uid: target.uid.as_deref(),
                        reference_kind: "k8s_object_reference",
                    },
                )?;
                insert_relation(
                    &mut relations,
                    structural_relation(
                        endpoint_entity.entity.clone(),
                        target_entity,
                        RelationKind::Other("Targets".into()),
                        collected_at_unix_ms,
                        "endpoint_target_ref",
                        base.context().source_confidence,
                    ),
                );
            }

            if let Some(node_name) = endpoint.node_name.as_deref() {
                let node = resolve_or_reference(
                    base.topology(),
                    &mut entities,
                    &cluster_namespace,
                    ReferenceSpec {
                        kind: "Node",
                        namespace: None,
                        name: node_name,
                        uid: None,
                        reference_kind: "k8s_node_reference",
                    },
                )?;
                insert_relation(
                    &mut relations,
                    structural_relation(
                        endpoint_entity.entity.clone(),
                        node,
                        RelationKind::HostedOn,
                        collected_at_unix_ms,
                        "endpoint_node_name",
                        base.context().source_confidence,
                    ),
                );
            }
        }
    }

    let snapshot = DiscoverySnapshot {
        integration_id: KUBERNETES_INTEGRATION_ID.into(),
        discovered_at_unix_ms: collected_at_unix_ms,
        entities: entities.into_values().collect(),
        relations: relations.into_values().collect(),
    };
    snapshot
        .validate()
        .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
    Ok(snapshot)
}

#[derive(Debug, Clone)]
struct ParsedSlice {
    name: String,
    namespace: String,
    uid: Option<String>,
    service_name: Option<String>,
    managed_by: Option<String>,
    address_type: AddressType,
    ports: Vec<EndpointPort>,
    endpoints: Vec<Endpoint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AddressType {
    Ipv4,
    Ipv6,
    Fqdn,
}

impl AddressType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ipv4 => "IPv4",
            Self::Ipv6 => "IPv6",
            Self::Fqdn => "FQDN",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct EndpointPort {
    name: String,
    protocol: String,
    port: Option<u16>,
    app_protocol: Option<String>,
}

#[derive(Debug, Clone)]
struct Endpoint {
    addresses: Vec<String>,
    hostname: Option<String>,
    node_name: Option<String>,
    zone: Option<String>,
    target_ref: Option<ObjectReference>,
    ready: ConditionValue,
    serving: ConditionValue,
    terminating: ConditionValue,
    hint_zones: Vec<String>,
    hint_nodes: Vec<String>,
}

#[derive(Debug, Clone)]
struct ObjectReference {
    kind: String,
    name: String,
    namespace: Option<String>,
    uid: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct ConditionValue {
    effective: bool,
    explicit: bool,
}

fn parse_slice(value: &Value) -> Result<ParsedSlice, IntegrationError> {
    let object = value.as_object().ok_or_else(|| {
        IntegrationError::Protocol("EndpointSlice replay document is not a JSON object".into())
    })?;
    let api_version = required_string(object.get("apiVersion"), "apiVersion")?;
    let kind = required_string(object.get("kind"), "kind")?;
    if api_version != ENDPOINTSLICE_API_VERSION || kind != ENDPOINTSLICE_KIND {
        return Err(IntegrationError::Protocol(format!(
            "EndpointSlice replay accepts only `{ENDPOINTSLICE_API_VERSION}` `{ENDPOINTSLICE_KIND}`, got `{api_version}` `{kind}`"
        )));
    }
    let metadata = object
        .get("metadata")
        .and_then(Value::as_object)
        .ok_or_else(|| IntegrationError::Protocol("EndpointSlice requires metadata".into()))?;
    let name = required_string(metadata.get("name"), "metadata.name")?;
    let namespace = required_string(metadata.get("namespace"), "metadata.namespace")?;
    let uid = optional_string(metadata.get("uid"));
    let labels = string_map(metadata.get("labels"), "metadata.labels")?;
    let service_name = labels.get(ENDPOINTSLICE_SERVICE_LABEL).cloned();
    let managed_by = labels.get(ENDPOINTSLICE_MANAGED_BY_LABEL).cloned();

    let address_type = match required_string(object.get("addressType"), "addressType")?.as_str() {
        "IPv4" => AddressType::Ipv4,
        "IPv6" => AddressType::Ipv6,
        "FQDN" => AddressType::Fqdn,
        other => {
            return Err(IntegrationError::Protocol(format!(
                "unsupported EndpointSlice addressType `{other}`"
            )))
        }
    };

    let ports = parse_ports(object.get("ports"))?;
    let endpoints = parse_endpoints(object.get("endpoints"), address_type)?;

    Ok(ParsedSlice {
        name,
        namespace,
        uid,
        service_name,
        managed_by,
        address_type,
        ports,
        endpoints,
    })
}

fn parse_ports(value: Option<&Value>) -> Result<Vec<EndpointPort>, IntegrationError> {
    let Some(value) = value else {
        return Ok(vec![]);
    };
    if value.is_null() {
        return Ok(vec![]);
    }
    let values = value
        .as_array()
        .ok_or_else(|| IntegrationError::Protocol("EndpointSlice ports is not an array".into()))?;
    if values.len() > MAX_PORTS_PER_SLICE {
        return Err(IntegrationError::Protocol(format!(
            "EndpointSlice has {} ports; maximum is {MAX_PORTS_PER_SLICE}",
            values.len()
        )));
    }
    let mut ports = Vec::with_capacity(values.len());
    let mut names = BTreeSet::new();
    for value in values {
        let port = value
            .as_object()
            .ok_or_else(|| IntegrationError::Protocol("EndpointSlice port is not an object".into()))?;
        let name = optional_string(port.get("name")).unwrap_or_default();
        if !names.insert(name.clone()) {
            return Err(IntegrationError::Protocol(format!(
                "EndpointSlice contains duplicate port name `{name}`"
            )));
        }
        let protocol = optional_string(port.get("protocol")).unwrap_or_else(|| "TCP".into());
        if !matches!(protocol.as_str(), "TCP" | "UDP" | "SCTP") {
            return Err(IntegrationError::Protocol(format!(
                "EndpointSlice port protocol `{protocol}` is not TCP, UDP, or SCTP"
            )));
        }
        let port_number = match port.get("port") {
            None | Some(Value::Null) => None,
            Some(value) => {
                let value = value.as_i64().ok_or_else(|| {
                    IntegrationError::Protocol("EndpointSlice port number is not an integer".into())
                })?;
                if !(1..=65_535).contains(&value) {
                    return Err(IntegrationError::Protocol(format!(
                        "EndpointSlice port {value} is outside 1..=65535"
                    )));
                }
                Some(value as u16)
            }
        };
        ports.push(EndpointPort {
            name,
            protocol,
            port: port_number,
            app_protocol: optional_string(port.get("appProtocol")),
        });
    }
    ports.sort();
    Ok(ports)
}

fn parse_endpoints(
    value: Option<&Value>,
    address_type: AddressType,
) -> Result<Vec<Endpoint>, IntegrationError> {
    let values = value
        .and_then(Value::as_array)
        .ok_or_else(|| IntegrationError::Protocol("EndpointSlice requires array endpoints".into()))?;
    if values.len() > MAX_ENDPOINTS_PER_SLICE {
        return Err(IntegrationError::Protocol(format!(
            "EndpointSlice has {} endpoints; maximum is {MAX_ENDPOINTS_PER_SLICE}",
            values.len()
        )));
    }

    values
        .iter()
        .map(|value| parse_endpoint(value, address_type))
        .collect()
}

fn parse_endpoint(value: &Value, address_type: AddressType) -> Result<Endpoint, IntegrationError> {
    let endpoint = value
        .as_object()
        .ok_or_else(|| IntegrationError::Protocol("EndpointSlice endpoint is not an object".into()))?;
    let addresses = endpoint
        .get("addresses")
        .and_then(Value::as_array)
        .ok_or_else(|| IntegrationError::Protocol("EndpointSlice endpoint requires addresses".into()))?;
    if addresses.is_empty() || addresses.len() > MAX_ADDRESSES_PER_ENDPOINT {
        return Err(IntegrationError::Protocol(format!(
            "EndpointSlice endpoint address count {} is outside 1..={MAX_ADDRESSES_PER_ENDPOINT}",
            addresses.len()
        )));
    }
    let mut parsed_addresses = addresses
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .ok_or_else(|| {
                    IntegrationError::Protocol(
                        "EndpointSlice endpoint address must be a non-empty string".into(),
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    for address in &parsed_addresses {
        validate_address(address_type, address)?;
    }
    parsed_addresses.sort();
    parsed_addresses.dedup();

    let conditions = endpoint.get("conditions").and_then(Value::as_object);
    let ready = condition(conditions, "ready", true)?;
    let serving = condition(conditions, "serving", true)?;
    let terminating = condition(conditions, "terminating", false)?;

    let target_ref = endpoint
        .get("targetRef")
        .filter(|value| !value.is_null())
        .map(parse_target_ref)
        .transpose()?;
    let (hint_zones, hint_nodes) = parse_hints(endpoint.get("hints"))?;

    Ok(Endpoint {
        addresses: parsed_addresses,
        hostname: optional_string(endpoint.get("hostname")),
        node_name: optional_string(endpoint.get("nodeName")),
        zone: optional_string(endpoint.get("zone")),
        target_ref,
        ready,
        serving,
        terminating,
        hint_zones,
        hint_nodes,
    })
}

fn condition(
    conditions: Option<&Map<String, Value>>,
    field: &str,
    default: bool,
) -> Result<ConditionValue, IntegrationError> {
    match conditions.and_then(|conditions| conditions.get(field)) {
        None | Some(Value::Null) => Ok(ConditionValue {
            effective: default,
            explicit: false,
        }),
        Some(value) => value
            .as_bool()
            .map(|effective| ConditionValue {
                effective,
                explicit: true,
            })
            .ok_or_else(|| {
                IntegrationError::Protocol(format!(
                    "EndpointSlice conditions.{field} is not boolean/null"
                ))
            }),
    }
}

fn parse_target_ref(value: &Value) -> Result<ObjectReference, IntegrationError> {
    let target = value.as_object().ok_or_else(|| {
        IntegrationError::Protocol("EndpointSlice targetRef is not an object".into())
    })?;
    Ok(ObjectReference {
        kind: required_string(target.get("kind"), "targetRef.kind")?,
        name: required_string(target.get("name"), "targetRef.name")?,
        namespace: optional_string(target.get("namespace")),
        uid: optional_string(target.get("uid")),
    })
}

fn parse_hints(value: Option<&Value>) -> Result<(Vec<String>, Vec<String>), IntegrationError> {
    let Some(hints) = value.filter(|value| !value.is_null()) else {
        return Ok((vec![], vec![]));
    };
    let hints = hints
        .as_object()
        .ok_or_else(|| IntegrationError::Protocol("EndpointSlice hints is not an object".into()))?;
    Ok((
        parse_named_hints(hints.get("forZones"), "forZones")?,
        parse_named_hints(hints.get("forNodes"), "forNodes")?,
    ))
}

fn parse_named_hints(value: Option<&Value>, field: &str) -> Result<Vec<String>, IntegrationError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(vec![]);
    };
    let values = value.as_array().ok_or_else(|| {
        IntegrationError::Protocol(format!("EndpointSlice hints.{field} is not an array"))
    })?;
    if values.len() > MAX_HINTS_PER_DIMENSION {
        return Err(IntegrationError::Protocol(format!(
            "EndpointSlice hints.{field} has {} entries; maximum is {MAX_HINTS_PER_DIMENSION}",
            values.len()
        )));
    }
    let mut result = values
        .iter()
        .map(|value| {
            value
                .as_object()
                .ok_or_else(|| {
                    IntegrationError::Protocol(format!(
                        "EndpointSlice hints.{field} entry is not an object"
                    ))
                })
                .and_then(|entry| required_string(entry.get("name"), "hints.name"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    result.sort();
    result.dedup();
    Ok(result)
}

fn validate_address(address_type: AddressType, address: &str) -> Result<(), IntegrationError> {
    match address_type {
        AddressType::Ipv4 => match address.parse::<IpAddr>() {
            Ok(IpAddr::V4(_)) => Ok(()),
            _ => Err(IntegrationError::Protocol(format!(
                "EndpointSlice IPv4 address `{address}` is not IPv4"
            ))),
        },
        AddressType::Ipv6 => match address.parse::<IpAddr>() {
            Ok(IpAddr::V6(_)) => Ok(()),
            _ => Err(IntegrationError::Protocol(format!(
                "EndpointSlice IPv6 address `{address}` is not IPv6"
            ))),
        },
        // Kubernetes documents no defined address semantics for FQDN slices.
        // Preserve the string without pretending it is routable or canonical IP.
        AddressType::Fqdn => Ok(()),
    }
}

fn endpoint_slice_entity(
    base: &KubernetesReplayDiscoverer,
    cluster_namespace: &str,
    slice: &ParsedSlice,
    _collected_at_unix_ms: u64,
) -> DiscoveredEntity {
    let id = slice.uid.clone().unwrap_or_else(|| {
        stable_id(&[
            "symthaea-kubernetes-name-v1",
            &base.context().cluster_id,
            ENDPOINTSLICE_API_VERSION,
            ENDPOINTSLICE_KIND,
            &slice.namespace,
            &slice.name,
        ])
    });
    let entity = EntityRef::new(cluster_namespace, "k8s_endpointslice", id);
    let mut attributes = BTreeMap::from([
        ("k8s.api_version".into(), ENDPOINTSLICE_API_VERSION.into()),
        ("k8s.kind".into(), ENDPOINTSLICE_KIND.into()),
        ("k8s.name".into(), slice.name.clone()),
        ("k8s.namespace".into(), slice.namespace.clone()),
        ("k8s.endpointslice.address_type".into(), slice.address_type.as_str().into()),
        ("k8s.endpointslice.endpoint_count".into(), slice.endpoints.len().to_string()),
        ("k8s.endpointslice.port_count".into(), slice.ports.len().to_string()),
    ]);
    if let Some(uid) = &slice.uid {
        attributes.insert("k8s.uid".into(), uid.clone());
    } else {
        attributes.insert("k8s.identity_quality".into(), "name_fallback".into());
    }
    if let Some(service) = &slice.service_name {
        attributes.insert(ENDPOINTSLICE_SERVICE_LABEL.into(), service.clone());
    }
    if let Some(managed_by) = &slice.managed_by {
        attributes.insert(ENDPOINTSLICE_MANAGED_BY_LABEL.into(), managed_by.clone());
    }
    if slice.address_type == AddressType::Fqdn {
        attributes.insert(
            "k8s.endpointslice.address_semantics".into(),
            "deprecated_undefined".into(),
        );
    }
    for (index, port) in slice.ports.iter().enumerate() {
        attributes.insert(format!("k8s.endpointslice.port.{index}.name"), port.name.clone());
        attributes.insert(
            format!("k8s.endpointslice.port.{index}.protocol"),
            port.protocol.clone(),
        );
        if let Some(number) = port.port {
            attributes.insert(
                format!("k8s.endpointslice.port.{index}.port"),
                number.to_string(),
            );
        }
        if let Some(app_protocol) = &port.app_protocol {
            attributes.insert(
                format!("k8s.endpointslice.port.{index}.app_protocol"),
                app_protocol.clone(),
            );
        }
    }
    DiscoveredEntity {
        entity,
        display_name: Some(slice.name.clone()),
        attributes,
    }
}

fn endpoint_membership_entity(
    cluster_namespace: &str,
    slice_entity: &EntityRef,
    slice: &ParsedSlice,
    endpoint: &Endpoint,
) -> DiscoveredEntity {
    let mut identity_parts = vec![
        "symthaea-kubernetes-endpoint-membership-v1".to_string(),
        slice_entity.canonical_key(),
        slice.address_type.as_str().to_string(),
    ];
    identity_parts.extend(endpoint.addresses.iter().cloned());
    if let Some(target) = &endpoint.target_ref {
        identity_parts.push(target.kind.clone());
        identity_parts.push(target.namespace.clone().unwrap_or_default());
        identity_parts.push(target.name.clone());
        identity_parts.push(target.uid.clone().unwrap_or_default());
    }
    identity_parts.push(endpoint.hostname.clone().unwrap_or_default());
    identity_parts.push(endpoint.node_name.clone().unwrap_or_default());
    identity_parts.push(endpoint.zone.clone().unwrap_or_default());
    let refs = identity_parts.iter().map(String::as_str).collect::<Vec<_>>();
    let id = format!("endpoint:{}", stable_hash(&refs));
    let entity = EntityRef::new(cluster_namespace, "k8s_endpoint_membership", id);

    let mut attributes = BTreeMap::from([
        ("k8s.endpoint.address_type".into(), slice.address_type.as_str().into()),
        ("k8s.endpoint.ready".into(), endpoint.ready.effective.to_string()),
        ("k8s.endpoint.ready.explicit".into(), endpoint.ready.explicit.to_string()),
        ("k8s.endpoint.serving".into(), endpoint.serving.effective.to_string()),
        (
            "k8s.endpoint.serving.explicit".into(),
            endpoint.serving.explicit.to_string(),
        ),
        (
            "k8s.endpoint.terminating".into(),
            endpoint.terminating.effective.to_string(),
        ),
        (
            "k8s.endpoint.terminating.explicit".into(),
            endpoint.terminating.explicit.to_string(),
        ),
    ]);
    for (index, address) in endpoint.addresses.iter().enumerate() {
        attributes.insert(format!("k8s.endpoint.address.{index}"), address.clone());
    }
    if endpoint.addresses.len() > 1 {
        attributes.insert(
            "k8s.endpoint.additional_address_semantics".into(),
            "undefined_by_kubernetes".into(),
        );
    }
    if let Some(hostname) = &endpoint.hostname {
        attributes.insert("k8s.endpoint.hostname".into(), hostname.clone());
    }
    if let Some(node_name) = &endpoint.node_name {
        attributes.insert("k8s.endpoint.node_name".into(), node_name.clone());
    }
    if let Some(zone) = &endpoint.zone {
        attributes.insert("k8s.endpoint.zone".into(), zone.clone());
    }
    for (index, zone) in endpoint.hint_zones.iter().enumerate() {
        attributes.insert(format!("k8s.endpoint.hint.zone.{index}"), zone.clone());
    }
    for (index, node) in endpoint.hint_nodes.iter().enumerate() {
        attributes.insert(format!("k8s.endpoint.hint.node.{index}"), node.clone());
    }
    if let Some(target) = &endpoint.target_ref {
        attributes.insert("k8s.endpoint.target.kind".into(), target.kind.clone());
        attributes.insert("k8s.endpoint.target.name".into(), target.name.clone());
        if let Some(namespace) = &target.namespace {
            attributes.insert("k8s.endpoint.target.namespace".into(), namespace.clone());
        }
        if let Some(uid) = &target.uid {
            attributes.insert("k8s.endpoint.target.uid".into(), uid.clone());
        }
    }

    DiscoveredEntity {
        entity,
        display_name: endpoint
            .hostname
            .clone()
            .or_else(|| endpoint.addresses.first().cloned()),
        attributes,
    }
}

struct ReferenceSpec<'a> {
    kind: &'a str,
    namespace: Option<&'a str>,
    name: &'a str,
    uid: Option<&'a str>,
    reference_kind: &'a str,
}

fn resolve_or_reference(
    base: &DiscoverySnapshot,
    entities: &mut BTreeMap<String, DiscoveredEntity>,
    cluster_namespace: &str,
    spec: ReferenceSpec<'_>,
) -> Result<EntityRef, IntegrationError> {
    if let Some(uid) = spec.uid {
        if let Some(entity) = base.entities.iter().find(|entity| {
            entity.attributes.get("k8s.uid").map(String::as_str) == Some(uid)
        }) {
            return Ok(entity.entity.clone());
        }
    }
    if let Some(entity) = find_entity_by_kind_name(base, spec.kind, spec.namespace, spec.name) {
        return Ok(entity);
    }

    let id = format!(
        "ref:{}",
        stable_hash(&[
            "symthaea-kubernetes-reference-v1",
            spec.kind,
            spec.namespace.unwrap_or(""),
            spec.name,
            spec.uid.unwrap_or(""),
        ])
    );
    let entity = EntityRef::new(cluster_namespace, spec.reference_kind, id);
    let mut attributes = BTreeMap::from([
        ("k8s.reference".into(), "true".into()),
        ("k8s.kind".into(), spec.kind.into()),
        ("k8s.name".into(), spec.name.into()),
    ]);
    if let Some(namespace) = spec.namespace {
        attributes.insert("k8s.namespace".into(), namespace.into());
    }
    if let Some(uid) = spec.uid {
        attributes.insert("k8s.uid".into(), uid.into());
    }
    insert_entity(
        entities,
        DiscoveredEntity {
            entity: entity.clone(),
            display_name: Some(spec.name.into()),
            attributes,
        },
    )?;
    Ok(entity)
}

fn find_entity_by_kind_name(
    snapshot: &DiscoverySnapshot,
    kind: &str,
    namespace: Option<&str>,
    name: &str,
) -> Option<EntityRef> {
    snapshot
        .entities
        .iter()
        .find(|entity| {
            entity.attributes.get("k8s.kind").map(String::as_str) == Some(kind)
                && entity.attributes.get("k8s.name").map(String::as_str) == Some(name)
                && match namespace {
                    Some(namespace) => {
                        entity.attributes.get("k8s.namespace").map(String::as_str)
                            == Some(namespace)
                    }
                    None => true,
                }
        })
        .map(|entity| entity.entity.clone())
}

fn structural_relation(
    from: EntityRef,
    to: EntityRef,
    kind: RelationKind,
    observed_at_unix_ms: u64,
    relationship: &str,
    confidence: f32,
) -> EntityRelation {
    EntityRelation {
        from,
        to,
        kind,
        basis: RelationBasis::Structural,
        confidence,
        observed_at_unix_ms: Some(observed_at_unix_ms),
        evidence_observation_ids: vec![],
        attributes: BTreeMap::from([("k8s.relationship".into(), relationship.into())]),
    }
}

fn insert_entity(
    entities: &mut BTreeMap<String, DiscoveredEntity>,
    entity: DiscoveredEntity,
) -> Result<(), IntegrationError> {
    let key = entity.entity.canonical_key();
    match entities.get(&key) {
        Some(existing) if existing != &entity => Err(IntegrationError::InvalidOutput(format!(
            "EndpointSlice entity key collision for `{key}`"
        ))),
        Some(_) => Ok(()),
        None => {
            entities.insert(key, entity);
            Ok(())
        }
    }
}

fn insert_relation(relations: &mut BTreeMap<RelationKey, EntityRelation>, relation: EntityRelation) {
    relations.entry(relation_key(&relation)).or_insert(relation);
}

fn relation_key(relation: &EntityRelation) -> RelationKey {
    (
        relation.from.clone(),
        relation.to.clone(),
        relation.kind.clone(),
        relation.basis,
    )
}

fn expand_documents<'a>(documents: &'a [Value]) -> Result<Vec<&'a Value>, IntegrationError> {
    let mut output = Vec::new();
    for document in documents {
        let kind = document.get("kind").and_then(Value::as_str).unwrap_or("");
        if kind == "EndpointSliceList" {
            let api_version = document
                .get("apiVersion")
                .and_then(Value::as_str)
                .unwrap_or("");
            if api_version != ENDPOINTSLICE_API_VERSION {
                return Err(IntegrationError::Protocol(format!(
                    "EndpointSliceList apiVersion must be `{ENDPOINTSLICE_API_VERSION}`, got `{api_version}`"
                )));
            }
            let items = document
                .get("items")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    IntegrationError::Protocol("EndpointSliceList requires array items".into())
                })?;
            output.extend(items.iter());
        } else {
            output.push(document);
        }
    }
    Ok(output)
}

fn required_string(value: Option<&Value>, field: &str) -> Result<String, IntegrationError> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| IntegrationError::Protocol(format!("EndpointSlice requires `{field}`")))
}

fn optional_string(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn string_map(
    value: Option<&Value>,
    field: &str,
) -> Result<BTreeMap<String, String>, IntegrationError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(BTreeMap::new());
    };
    let map = value.as_object().ok_or_else(|| {
        IntegrationError::Protocol(format!("EndpointSlice `{field}` is not an object"))
    })?;
    map.iter()
        .map(|(key, value)| {
            value
                .as_str()
                .map(|value| (key.clone(), value.to_string()))
                .ok_or_else(|| {
                    IntegrationError::Protocol(format!(
                        "EndpointSlice `{field}.{key}` is not a string"
                    ))
                })
        })
        .collect()
}

fn stable_id(parts: &[&str]) -> String {
    format!("name:{}", stable_hash(parts))
}

fn stable_hash(parts: &[&str]) -> String {
    let mut hasher = Hasher::new();
    for value in parts {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;
    use symthaea_integration_core::{IntegrationId, IntegrationRegistry};
    use symthaea_kubernetes_bridge::KubernetesReplayContext;

    fn base(at: u64) -> KubernetesReplayDiscoverer {
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
                    "apiVersion":"v1","kind":"Pod",
                    "metadata":{"name":"api-pod","namespace":"shop","uid":"pod-1"}
                }),
                json!({
                    "apiVersion":"v1","kind":"Service",
                    "metadata":{"name":"api","namespace":"shop","uid":"svc-1"}
                }),
            ],
            at,
        )
        .unwrap()
    }

    fn slice() -> Value {
        json!({
            "apiVersion":"discovery.k8s.io/v1",
            "kind":"EndpointSlice",
            "metadata":{
                "name":"api-abc",
                "namespace":"shop",
                "uid":"slice-1",
                "labels":{
                    "kubernetes.io/service-name":"api",
                    "endpointslice.kubernetes.io/managed-by":"endpointslice-controller.k8s.io"
                }
            },
            "addressType":"IPv4",
            "ports":[{"name":"http","protocol":"TCP","port":8080}],
            "endpoints":[{
                "addresses":["10.1.2.3"],
                "conditions":{"ready":true,"serving":true,"terminating":false},
                "nodeName":"node-a",
                "zone":"zone-a",
                "targetRef":{
                    "kind":"Pod","name":"api-pod","namespace":"shop","uid":"pod-1"
                },
                "hints":{"forZones":[{"name":"zone-a"}]}
            }]
        })
    }

    #[test]
    fn endpoint_slice_adds_service_endpoint_target_and_node_topology() {
        let base = base(100);
        let snapshot = augment_endpoint_slices(&base, &[slice()], 100).unwrap();
        snapshot.validate().unwrap();
        assert!(snapshot.entities.iter().any(|entity| entity.entity.kind == "k8s_endpointslice"));
        assert!(snapshot.entities.iter().any(|entity| entity.entity.kind == "k8s_endpoint_membership"));
        assert!(snapshot.relations.iter().any(|relation| {
            relation.kind == RelationKind::Other("EndpointSliceFor".into())
        }));
        assert!(snapshot.relations.iter().any(|relation| {
            relation.kind == RelationKind::Other("Targets".into())
                && relation.to.kind == "k8s_pod"
        }));
        assert!(snapshot.relations.iter().any(|relation| {
            relation.kind == RelationKind::HostedOn && relation.to.kind == "k8s_node"
        }));
    }

    #[test]
    fn nil_conditions_preserve_default_and_presence_semantics() {
        let base = base(100);
        let mut slice = slice();
        slice["endpoints"][0]["conditions"] = json!({});
        let snapshot = augment_endpoint_slices(&base, &[slice], 100).unwrap();
        let endpoint = snapshot
            .entities
            .iter()
            .find(|entity| entity.entity.kind == "k8s_endpoint_membership")
            .unwrap();
        assert_eq!(endpoint.attributes.get("k8s.endpoint.ready").map(String::as_str), Some("true"));
        assert_eq!(endpoint.attributes.get("k8s.endpoint.ready.explicit").map(String::as_str), Some("false"));
        assert_eq!(endpoint.attributes.get("k8s.endpoint.serving").map(String::as_str), Some("true"));
        assert_eq!(endpoint.attributes.get("k8s.endpoint.terminating").map(String::as_str), Some("false"));
    }

    #[test]
    fn different_capture_time_is_rejected() {
        let base = base(100);
        assert!(augment_endpoint_slices(&base, &[slice()], 101).is_err());
    }

    #[test]
    fn invalid_address_family_is_rejected() {
        let base = base(100);
        let mut slice = slice();
        slice["endpoints"][0]["addresses"] = json!(["2001:db8::1"]);
        assert!(augment_endpoint_slices(&base, &[slice], 100).is_err());
    }

    #[test]
    fn missing_service_or_target_becomes_reference_not_fake_canonical_entity() {
        let base = KubernetesReplayDiscoverer::from_objects(
            KubernetesReplayContext::default(),
            &[json!({
                "apiVersion":"v1","kind":"Namespace",
                "metadata":{"name":"shop","uid":"ns-1"}
            })],
            100,
        )
        .unwrap();
        let snapshot = augment_endpoint_slices(&base, &[slice()], 100).unwrap();
        assert!(snapshot.entities.iter().any(|entity| entity.entity.kind == "k8s_service_reference"));
        assert!(snapshot.entities.iter().any(|entity| entity.entity.kind == "k8s_object_reference"));
    }

    #[test]
    fn augmented_topology_passes_existing_registry_budget_boundary() {
        let base = Arc::new(base(100));
        let snapshot = augment_endpoint_slices(base.as_ref(), &[slice()], 100).unwrap();
        let mut registry = IntegrationRegistry::new();
        registry.register_discoverer(base).unwrap();
        registry
            .admit_discovery_snapshot(
                &IntegrationId::new(KUBERNETES_INTEGRATION_ID),
                &snapshot,
            )
            .unwrap();
    }

    #[test]
    fn endpointslice_list_is_expanded() {
        let base = base(100);
        let list = json!({
            "apiVersion":"discovery.k8s.io/v1",
            "kind":"EndpointSliceList",
            "items":[slice()]
        });
        let snapshot = augment_endpoint_slices(&base, &[list], 100).unwrap();
        assert_eq!(
            snapshot
                .entities
                .iter()
                .filter(|entity| entity.entity.kind == "k8s_endpointslice")
                .count(),
            1
        );
    }
}
