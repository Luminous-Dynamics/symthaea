// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit standards normalization for identity evidence.
//!
//! Normalization is deliberately a named step rather than a hidden resolver
//! alias. The resolver only compares canonical identifiers it is given.

use crate::{
    DiscoverySnapshot, IdentitySnapshot, IdentitySnapshotError, IdentityValidationError,
    kubernetes_object_uid_identifier,
};
use std::collections::BTreeSet;

/// Transitional scheme emitted by the first Kubernetes replay tranche before
/// the shared semantic-identity vocabulary existed.
pub const LEGACY_K8S_UID_SCHEME: &str = "k8s.uid";

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum IdentityNormalizationError {
    #[error("identity normalization received invalid identity evidence: {0}")]
    InvalidIdentity(#[from] IdentityValidationError),
    #[error("normalized identity snapshot is invalid: {0}")]
    InvalidSnapshot(#[from] IdentitySnapshotError),
    #[error("topology contains conflicting kube-system namespace UIDs: {0:?}")]
    ConflictingKubernetesClusterUids(Vec<String>),
}

/// Extract the OpenTelemetry-compatible Kubernetes cluster pseudo-ID from an
/// admitted/replayed topology when the kube-system Namespace object is present.
///
/// OpenTelemetry currently defines `k8s.cluster.uid` as the UID of the
/// `kube-system` Namespace. A caller-supplied cluster display name is not a
/// substitute for this evidence.
pub fn kubernetes_cluster_uid_from_topology(
    topology: &DiscoverySnapshot,
) -> Result<Option<String>, IdentityNormalizationError> {
    let mut candidates = BTreeSet::new();
    for entity in &topology.entities {
        if entity.entity.kind != "k8s_namespace" {
            continue;
        }
        if entity.attributes.get("k8s.name").map(String::as_str) != Some("kube-system") {
            continue;
        }
        if let Some(uid) = entity
            .attributes
            .get("k8s.uid")
            .map(String::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            candidates.insert(uid.to_string());
        }
    }

    match candidates.len() {
        0 => Ok(None),
        1 => Ok(candidates.into_iter().next()),
        _ => Err(IdentityNormalizationError::ConflictingKubernetesClusterUids(
            candidates.into_iter().collect(),
        )),
    }
}

/// Upgrade legacy Kubernetes object UID claims to the shared semantic scheme
/// using a *real* Kubernetes cluster UID.
///
/// This function intentionally requires the caller to provide that UID. It will
/// not derive strong scope from a cluster name, tenant, hostname, or adapter
/// namespace. Claims already using another scheme are preserved unchanged.
pub fn normalize_kubernetes_uid_snapshot(
    snapshot: &IdentitySnapshot,
    cluster_uid: &str,
) -> Result<IdentitySnapshot, IdentityNormalizationError> {
    let mut normalized = snapshot.clone();
    for claim in &mut normalized.claims {
        if claim.identifier.scheme != LEGACY_K8S_UID_SCHEME {
            continue;
        }

        claim.identifier = kubernetes_object_uid_identifier(
            &claim.subject.kind,
            &claim.identifier.value,
            Some(cluster_uid),
        )?;
        claim.claim_id = normalized_claim_id(
            &claim.source.integration_id,
            &claim.claim_id,
            &claim.identifier.canonical_key()?,
        );
    }
    normalized.validate()?;
    Ok(normalized)
}

fn normalized_claim_id(source: &str, original: &str, canonical_identifier: &str) -> String {
    let mut output = String::from("normalized-claim-v1");
    push(&mut output, source);
    push(&mut output, original);
    push(&mut output, canonical_identifier);
    output
}

fn push(output: &mut String, value: &str) {
    output.push('|');
    output.push_str(&value.len().to_string());
    output.push(':');
    output.push_str(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DiscoveredEntity, EntityRef, ExternalIdentifier, IdentifierStability,
        IdentifierUniqueness, IdentityClaim, IdentityClaimSource, IdentityStrength,
    };
    use std::collections::BTreeMap;

    fn legacy_snapshot() -> IdentitySnapshot {
        IdentitySnapshot {
            integration_id: "kubernetes-object-replay".into(),
            collected_at_unix_ms: 10,
            claims: vec![IdentityClaim {
                claim_id: "legacy-pod-claim".into(),
                subject: EntityRef::new("k8s:fixture", "k8s_pod", "pod-uid"),
                identifier: ExternalIdentifier {
                    scheme: LEGACY_K8S_UID_SCHEME.into(),
                    value: "pod-uid".into(),
                    scope: Some("legacy-cluster-name-scope".into()),
                    uniqueness: IdentifierUniqueness::Scoped,
                    stability: IdentifierStability::Persistent,
                    case_sensitive: true,
                },
                strength: IdentityStrength::Strong,
                source_confidence: 1.0,
                source: IdentityClaimSource {
                    integration_id: "kubernetes-object-replay".into(),
                    collector_id: None,
                    tenant: None,
                },
                observed_at_unix_ms: 10,
                valid_from_unix_ms: None,
                valid_until_unix_ms: None,
                evidence_observation_ids: vec![],
            }],
            separation_claims: vec![],
        }
    }

    #[test]
    fn normalization_replaces_legacy_name_scope_with_real_cluster_uid_scope() {
        let normalized = normalize_kubernetes_uid_snapshot(&legacy_snapshot(), "cluster-uid").unwrap();
        let claim = &normalized.claims[0];
        assert_eq!(claim.identifier.scheme, "k8s.pod.uid");
        assert!(claim.identifier.scope.as_deref().unwrap().contains("cluster-uid"));
        assert!(claim.claim_id.starts_with("normalized-claim-v1"));
    }

    #[test]
    fn kube_system_namespace_uid_is_extracted_as_cluster_uid() {
        let topology = DiscoverySnapshot {
            integration_id: "fixture".into(),
            discovered_at_unix_ms: 10,
            entities: vec![DiscoveredEntity {
                entity: EntityRef::new("k8s:fixture", "k8s_namespace", "cluster-uid"),
                display_name: Some("kube-system".into()),
                attributes: BTreeMap::from([
                    ("k8s.name".into(), "kube-system".into()),
                    ("k8s.uid".into(), "cluster-uid".into()),
                ]),
            }],
            relations: vec![],
        };
        assert_eq!(
            kubernetes_cluster_uid_from_topology(&topology).unwrap(),
            Some("cluster-uid".into())
        );
    }

    #[test]
    fn cluster_name_alone_never_becomes_cluster_uid() {
        let topology = DiscoverySnapshot {
            integration_id: "fixture".into(),
            discovered_at_unix_ms: 10,
            entities: vec![DiscoveredEntity {
                entity: EntityRef::new("k8s:fixture", "k8s_cluster", "prod"),
                display_name: Some("prod".into()),
                attributes: BTreeMap::from([("k8s.cluster_id".into(), "prod".into())]),
            }],
            relations: vec![],
        };
        assert_eq!(kubernetes_cluster_uid_from_topology(&topology).unwrap(), None);
    }
}
