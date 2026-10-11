// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use blake3::Hasher;
use serde_json::{Map, Value, json};
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
            "EndpointSlice capture time {collected_at_unix_ms} does not match base capture time {}",
            base.topology().discovered_at_unix_ms
        )));
    }

    let cluster = base
        .topology()
        .entities
        .iter()
        .find(|entity| entity.entity.kind == "k8s_cluster")
        .cloned()
        .ok_or_else(|| {
            IntegrationError::InvalidOutput("base Kubernetes topology has no cluster entity".into())
        })?;
    let cluster_namespace = cluster.entity.namespace.clone();

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
    let mut seen_slices = BTreeSet::new();

    for document in expand_documents(documents)? {
        let slice = parse_slice(document)?;
        let slice_entity = build_slice_entity(base, &cluster_namespace, &slice)?;
        let slice_key = slice_entity.entity.canonical_key();
        if !seen_slices.insert(slice_key) {
            return Err(IntegrationError::Protocol(format!(
                "duplicate EndpointSlice replay object `{}`",
                slice.name
            )));
        }
        merge_entity(&mut entities, slice_entity.clone())?;

        let namespace = resolve_reference(
            base.topology(),
            &mut entities,
            &cluster_namespace,
            ReferenceSpec {
                api_version: Some("v1"),
                kind: "Namespace",
                namespace: None,
                name: &slice.namespace,
                uid: None,
                role: "namespace_reference",
            },
        )?;
        insert_relation(
            &mut relations,
            structural_relation(
                slice_entity.entity.clone(),
                namespace.clone(),
                RelationKind::MemberOf,
                collected_at_unix_ms,
                "endpointslice_namespace_membership",
                base.context().source_confidence,
            ),
        );
        insert_relation(
            &mut relations,
            structural_relation(
                namespace,
                cluster.entity.clone(),
                RelationKind::MemberOf,
                collected_at_unix_ms,
                "namespace_cluster_membership",
                base.context().source_confidence,
            ),
        );

        let service = if let Some(service_name) = slice.service_name.as_deref() {
            let service = resolve_reference(
                base.topology(),
                &mut entities,
                &cluster_namespace,
                ReferenceSpec {
                    api_version: Some("v1"),
                    kind: "Service",
                    namespace: Some(&slice.namespace),
                    name: service_name,
                    uid: None,
                    role: "service_reference",
                },
            )?;
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
            Some(service)
        } else {
            None
        };

        let mut seen_endpoints = BTreeSet::new();
        for endpoint in &slice.endpoints {
            let endpoint_entity = build_endpoint_entity(
                &cluster_namespace,
                &slice_entity.entity,
                &slice,
                endpoint,
            )?;
            if !seen_endpoints.insert(endpoint_entity.entity.canonical_key()) {
                return Err(IntegrationError::Protocol(format!(
                    "EndpointSlice `{}` contains duplicate endpoint memberships",
                    slice.name
                )));
            }
            merge_entity(&mut entities, endpoint_entity.clone())?;
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

            if let Some(service) = &service {
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
                let target_entity = resolve_reference(
                    base.topology(),
                    &mut entities,
                    &cluster_namespace,
                    ReferenceSpec {
                        api_version: target.api_version.as_deref(),
                        kind: &target.kind,
                        namespace: target.namespace.as_deref(),
                        name: &target.name,
                        uid: target.uid.as_deref(),
                        role: "target_reference",
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
                let node = resolve_reference(
                    base.topology(),
                    &mut entities,
                    &cluster_namespace,
                    ReferenceSpec {
                        api_version: Some("v1"),
                        kind: "Node",
                        namespace: None,
                        name: node_name,
                        uid: None,
                        role: "node_reference",
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
    protocol_explicit: bool,
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
    api_version: Option<String>,
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
        IntegrationError::Protocol("EndpointSlice replay document is not an object".into())
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

    Ok(ParsedSlice {
        name,
        namespace,
        uid,
        service_name,
        managed_by,
        address_type,
        ports: parse_ports(object.get("ports"))?,
        endpoints: parse_endpoints(object.get("endpoints"), address_type)?,
    })
}

fn parse_ports(value: Option<&Value>) -> Result<Vec<EndpointPort>, IntegrationError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(vec![]);
    };
    let values = value
        .as_array()
        .ok_or_else(|| IntegrationError::Protocol("EndpointSlice ports is not an array".into()))?;
    if values.len() > MAX_PORTS_PER_SLICE {
        return Err(IntegrationError::Protocol(format!(
            "EndpointSlice has {} ports; maximum is {MAX_PORTS_PER_SLICE}",
            values.len()
        )));
    }

    let mut names = BTreeSet::new();
    let mut ports = Vec::with_capacity(values.len());
    for value in values {
        let value = value
            .as_object()
            .ok_or_else(|| IntegrationError::Protocol("EndpointSlice port is not an object".into()))?;
        let name = optional_string(value.get("name")).unwrap_or_default();
        if !names.insert(name.clone()) {
            return Err(IntegrationError::Protocol(format!(
                "EndpointSlice contains duplicate port name `{name}`"
            )));
        }
        let protocol_value = optional_string(value.get("protocol"));
        let protocol_explicit = protocol_value.is_some();
        let protocol = protocol_value.unwrap_or_else(|| "TCP".into());
        if !matches!(protocol.as_str(), "TCP" | "UDP" | "SCTP") {
            return Err(IntegrationError::Protocol(format!(
                "EndpointSlice protocol `{protocol}` is not TCP, UDP, or SCTP"
            )));
        }
        let port = match value.get("port") {
            None | Some(Value::Null) => None,
            Some(value) => {
                let number = value.as_i64().ok_or_else(|| {
                    IntegrationError::Protocol("EndpointSlice port is not an integer".into())
                })?;
                if !(1..=65_535).contains(&number) {
                    return Err(IntegrationError::Protocol(format!(
                        "EndpointSlice port {number} is outside 1..=65535"
                    )));
                }
                Some(number as u16)
            }
        };
        ports.push(EndpointPort {
            name,
            protocol,
            protocol_explicit,
            port,
            app_protocol: optional_string(value.get("appProtocol")),
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
    let raw_addresses = endpoint
        .get("addresses")
        .and_then(Value::as_array)
        .ok_or_else(|| IntegrationError::Protocol("EndpointSlice endpoint requires addresses".into()))?;
    if raw_addresses.is_empty() || raw_addresses.len() > MAX_ADDRESSES_PER_ENDPOINT {
        return Err(IntegrationError::Protocol(format!(
            "EndpointSlice endpoint address count {} is outside 1..={MAX_ADDRESSES_PER_ENDPOINT}",
            raw_addresses.len()
        )));
    }
    let mut addresses = raw_addresses
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .ok_or_else(|| {
                    IntegrationError::Protocol(
                        "EndpointSlice address must be a non-empty string".into(),
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    for address in &addresses {
        validate_address(address_type, address)?;
    }
    addresses.sort();
    if addresses.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(IntegrationError::Protocol(
            "EndpointSlice endpoint contains duplicate addresses".into(),
        ));
    }

    let conditions = endpoint.get("conditions").and_then(Value::as_object);
    let target_ref = endpoint
        .get("targetRef")
        .filter(|value| !value.is_null())
        .map(parse_target_ref)
        .transpose()?;
    let (hint_zones, hint_nodes) = parse_hints(endpoint.get("hints"))?;
    Ok(Endpoint {
        addresses,
        hostname: optional_string(endpoint.get("hostname")),
        node_name: optional_string(endpoint.get("nodeName")),
        zone: optional_string(endpoint.get("zone")),
        target_ref,
        ready: condition(conditions, "ready", true)?,
        serving: condition(conditions, "serving", true)?,
        terminating: condition(conditions, "terminating", false)?,
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
        api_version: optional_string(target.get("apiVersion")),
        kind: required_string(target.get("kind"), "targetRef.kind")?,
        name: required_string(target.get("name"), "targetRef.name")?,
        namespace: optional_string(target.get("namespace")),
        uid: optional_string(target.get("uid")),
    })
}

fn parse_hints(value: Option<&Value>) -> Result<(Vec<String>, Vec<String>), IntegrationError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok((vec![], vec![]));
    };
    let hints = value
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
                .and_then(|value| required_string(value.get("name"), "hints.name"))
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
        // Kubernetes defines no routing semantics for FQDN EndpointSlice
        // addresses. Preserve the string but never interpret it as an IP.
        AddressType::Fqdn => Ok(()),
    }
}

fn build_slice_entity(
    base: &KubernetesReplayDiscoverer,
    cluster_namespace: &str,
    slice: &ParsedSlice,
) -> Result<DiscoveredEntity, IntegrationError> {
    let existing = if let Some(uid) = slice.uid.as_deref() {
        find_entity_by_uid_kind(base.topology(), uid, ENDPOINTSLICE_KIND)?
    } else {
        find_entity_exact(
            base.topology(),
            ENDPOINTSLICE_API_VERSION,
            ENDPOINTSLICE_KIND,
            Some(&slice.namespace),
            &slice.name,
        )
    };
    let entity = existing.unwrap_or_else(|| {
        let id = slice.uid.clone().unwrap_or_else(|| {
            stable_name_id(
                &base.context().cluster_id,
                ENDPOINTSLICE_API_VERSION,
                ENDPOINTSLICE_KIND,
                Some(&slice.namespace),
                &slice.name,
            )
        });
        EntityRef::new(cluster_namespace, "k8s_object", id)
    });

    let mut attributes = BTreeMap::from([
        ("k8s.api_version".into(), ENDPOINTSLICE_API_VERSION.into()),
        ("k8s.kind".into(), ENDPOINTSLICE_KIND.into()),
        ("k8s.name".into(), slice.name.clone()),
        ("k8s.namespace".into(), slice.namespace.clone()),
        ("symthaea.k8s.role".into(), "endpoint_slice".into()),
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
        let encoded = serde_json::to_string(&json!({
            "name": port.name,
            "protocol": port.protocol,
            "protocol_explicit": port.protocol_explicit,
            "port": port.port,
            "appProtocol": port.app_protocol,
        }))
        .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
        attributes.insert(format!("k8s.endpointslice.port.{index}"), encoded);
    }
    Ok(DiscoveredEntity {
        entity,
        display_name: Some(slice.name.clone()),
        attributes,
    })
}

fn build_endpoint_entity(
    cluster_namespace: &str,
    slice_entity: &EntityRef,
    slice: &ParsedSlice,
    endpoint: &Endpoint,
) -> Result<DiscoveredEntity, IntegrationError> {
    let mut parts = vec![
        "symthaea-kubernetes-endpoint-membership-v1".to_string(),
        slice_entity.canonical_key(),
        slice.address_type.as_str().to_string(),
    ];
    parts.extend(endpoint.addresses.iter().cloned());
    if let Some(target) = &endpoint.target_ref {
        parts.push(target.api_version.clone().unwrap_or_default());
        parts.push(target.kind.clone());
        parts.push(target.namespace.clone().unwrap_or_default());
        parts.push(target.name.clone());
        parts.push(target.uid.clone().unwrap_or_default());
    }
    parts.push(endpoint.hostname.clone().unwrap_or_default());
    parts.push(endpoint.node_name.clone().unwrap_or_default());
    parts.push(endpoint.zone.clone().unwrap_or_default());
    let refs = parts.iter().map(String::as_str).collect::<Vec<_>>();
    let entity = EntityRef::new(
        cluster_namespace,
        "k8s_object",
        format!("endpoint:{}", stable_hash(&refs)),
    );

    let addresses_json = serde_json::to_string(&endpoint.addresses)
        .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
    let hint_zones_json = serde_json::to_string(&endpoint.hint_zones)
        .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
    let hint_nodes_json = serde_json::to_string(&endpoint.hint_nodes)
        .map_err(|error| IntegrationError::InvalidOutput(error.to_string()))?;
    let mut attributes = BTreeMap::from([
        ("symthaea.k8s.role".into(), "endpoint_membership".into()),
        ("k8s.endpoint.address_type".into(), slice.address_type.as_str().into()),
        ("k8s.endpoint.addresses".into(), addresses_json),
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
        ("k8s.endpoint.hint_zones".into(), hint_zones_json),
        ("k8s.endpoint.hint_nodes".into(), hint_nodes_json),
    ]);
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
    if let Some(target) = &endpoint.target_ref {
        if let Some(api_version) = &target.api_version {
            attributes.insert("k8s.endpoint.target.api_version".into(), api_version.clone());
        }
        attributes.insert("k8s.endpoint.target.kind".into(), target.kind.clone());
        attributes.insert("k8s.endpoint.target.name".into(), target.name.clone());
        if let Some(namespace) = &target.namespace {
            attributes.insert("k8s.endpoint.target.namespace".into(), namespace.clone());
        }
        if let Some(uid) = &target.uid {
            attributes.insert("k8s.endpoint.target.uid".into(), uid.clone());
        }
    }
    Ok(DiscoveredEntity {
        entity,
        display_name: endpoint
            .hostname
            .clone()
            .or_else(|| endpoint.addresses.first().cloned()),
        attributes,
    })
}

struct ReferenceSpec<'a> {
    api_version: Option<&'a str>,
    kind: &'a str,
    namespace: Option<&'a str>,
    name: &'a str,
    uid: Option<&'a str>,
    role: &'a str,
}

fn resolve_reference(
    base: &DiscoverySnapshot,
    entities: &mut BTreeMap<String, DiscoveredEntity>,
    cluster_namespace: &str,
    spec: ReferenceSpec<'_>,
) -> Result<EntityRef, IntegrationError> {
    if let Some(uid) = spec.uid {
        if let Some(entity) = base
            .entities
            .iter()
            .find(|entity| entity.attributes.get("k8s.uid").map(String::as_str) == Some(uid))
        {
            let actual_kind = entity.attributes.get("k8s.kind").map(String::as_str);
            if actual_kind != Some(spec.kind) {
                return Err(IntegrationError::InvalidOutput(format!(
                    "EndpointSlice target UID `{uid}` claims kind `{}` but base topology says `{:?}`",
                    spec.kind, actual_kind
                )));
            }
            return Ok(entity.entity.clone());
        }
    }

    if let Some(api_version) = spec.api_version {
        if let Some(entity) = find_entity_exact(
            base,
            api_version,
            spec.kind,
            spec.namespace,
            spec.name,
        ) {
            return Ok(entity);
        }
    }

    let entity = EntityRef::new(
        cluster_namespace,
        "k8s_object",
        format!(
            "ref:{}",
            stable_hash(&[
                "symthaea-kubernetes-reference-v1",
                spec.api_version.unwrap_or(""),
                spec.kind,
                spec.namespace.unwrap_or(""),
                spec.name,
                spec.uid.unwrap_or(""),
            ])
        ),
    );
    let mut attributes = BTreeMap::from([
        ("k8s.reference".into(), "true".into()),
        ("symthaea.k8s.role".into(), spec.role.into()),
        ("k8s.kind".into(), spec.kind.into()),
        ("k8s.name".into(), spec.name.into()),
    ]);
    if let Some(api_version) = spec.api_version {
        attributes.insert("k8s.api_version".into(), api_version.into());
    }
    if let Some(namespace) = spec.namespace {
        attributes.insert("k8s.namespace".into(), namespace.into());
    }
    if let Some(uid) = spec.uid {
        attributes.insert("k8s.uid".into(), uid.into());
    }
    merge_entity(
        entities,
        DiscoveredEntity {
            entity: entity.clone(),
            display_name: Some(spec.name.into()),
            attributes,
        },
    )?;
    Ok(entity)
}

fn find_entity_by_uid_kind(
    snapshot: &DiscoverySnapshot,
    uid: &str,
    kind: &str,
) -> Result<Option<EntityRef>, IntegrationError> {
    let Some(entity) = snapshot
        .entities
        .iter()
        .find(|entity| entity.attributes.get("k8s.uid").map(String::as_str) == Some(uid))
    else {
        return Ok(None);
    };
    let actual_kind = entity.attributes.get("k8s.kind").map(String::as_str);
    if actual_kind != Some(kind) {
        return Err(IntegrationError::InvalidOutput(format!(
            "Kubernetes UID `{uid}` is `{actual_kind:?}`, not `{kind}`"
        )));
    }
    Ok(Some(entity.entity.clone()))
}

fn find_entity_exact(
    snapshot: &DiscoverySnapshot,
    api_version: &str,
    kind: &str,
    namespace: Option<&str>,
    name: &str,
) -> Option<EntityRef> {
    snapshot
        .entities
        .iter()
        .find(|entity| {
            entity.attributes.get("k8s.api_version").map(String::as_str) == Some(api_version)
                && entity.attributes.get("k8s.kind").map(String::as_str) == Some(kind)
                && entity.attributes.get("k8s.name").map(String::as_str) == Some(name)
                && match namespace {
                    Some(namespace) => {
                        entity.attributes.get("k8s.namespace").map(String::as_str)
                            == Some(namespace)
                    }
                    None => !entity.attributes.contains_key("k8s.namespace"),
                }
        })
        .map(|entity| entity.entity.clone())
}

fn merge_entity(
    entities: &mut BTreeMap<String, DiscoveredEntity>,
    incoming: DiscoveredEntity,
) -> Result<(), IntegrationError> {
    let key = incoming.entity.canonical_key();
    let Some(existing) = entities.get_mut(&key) else {
        entities.insert(key, incoming);
        return Ok(());
    };
    if let (Some(existing_name), Some(incoming_name)) = (&existing.display_name, &incoming.display_name) {
        if existing_name != incoming_name {
            return Err(IntegrationError::InvalidOutput(format!(
                "EndpointSlice entity `{key}` has conflicting display names"
            )));
        }
    } else if existing.display_name.is_none() {
        existing.display_name = incoming.display_name;
    }
    for (attribute, value) in incoming.attributes {
        if let Some(existing_value) = existing.attributes.get(&attribute) {
            if existing_value != &value {
                return Err(IntegrationError::InvalidOutput(format!(
                    "EndpointSlice entity `{key}` has conflicting attribute `{attribute}`"
                )));
            }
        } else {
            existing.attributes.insert(attribute, value);
        }
    }
    Ok(())
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

fn string_map(value: Option<&Value>, field: &str) -> Result<BTreeMap<String, String>, IntegrationError> {
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

fn stable_name_id(
    cluster_id: &str,
    api_version: &str,
    kind: &str,
    namespace: Option<&str>,
    name: &str,
) -> String {
    format!(
        "name:{}",
        stable_hash(&[
            "symthaea-kubernetes-name-v1",
            cluster_id,
            api_version,
            kind,
            namespace.unwrap_or(""),
            name,
        ])
    )
}

fn stable_hash(parts: &[&str]) -> String {
    let mut hasher = Hasher::new();
    for value in parts {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    hasher.finalize().to_hex().to_string()
}
